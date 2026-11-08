use crate::misc::index::{CollectedIds, FUTURE_CHECK_CID};
use crate::{bin, jit};
use hashbrown::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::{RwLock, watch};
use tokio_util::sync::CancellationToken;

type InputVirtualCheckMap = Vec<Vec<jit::jit_gadget_type::Error>>;

fn input_virtual_check_map(
    gadget_type: &jit::JitGadgetType,
    port_types: &HashMap<u64, Arc<jit::JitPortType>>,
) -> Arc<InputVirtualCheckMap> {
    let mut input_map: InputVirtualCheckMap = gadget_type
        .base
        .as_ref()
        .unwrap()
        .inputs
        .iter()
        .map(|port| vec![jit::jit_gadget_type::Error::default(); port_types[&port.ptype].stabilizers.len()])
        .collect();
    for (finished, checks) in [(true, &gadget_type.finished_checks), (false, &gadget_type.unfinished_checks)] {
        for (index, check) in checks.iter().enumerate() {
            for measurement in &check.measurements {
                if let Some(port) = measurement.input_port {
                    let port = usize::try_from(port).expect("input port index exceeds usize");
                    let measurement_index =
                        usize::try_from(measurement.measurement_index).expect("input check index exceeds usize");
                    let entry = &mut input_map[port][measurement_index];
                    if finished {
                        entry.finished_checks.push(index as u64);
                    } else {
                        entry.unfinished_checks.push(index as u64);
                    }
                }
            }
        }
    }
    Arc::new(input_map)
}

pub struct JitCompiler {
    pub jit_port_types: RwLock<HashMap<u64, Arc<jit::JitPortType>>>,
    pub jit_gadget_types: RwLock<HashMap<u64, Arc<jit::JitGadgetType>>>,
    pub terminal_error_model_types: RwLock<HashMap<u64, Arc<bin::ErrorModelType>>>,
    input_virtual_check_maps: RwLock<HashMap<u64, Arc<InputVirtualCheckMap>>>,
    pub current_gid: AtomicU64,
    pub gadgets: RwLock<HashMap<u64, JitGadgetState>>,
    collected_gids: RwLock<CollectedIds>,
    collection_candidates: std::sync::Mutex<Vec<u64>>,
    error_templates: tokio::sync::Mutex<lru::LruCache<ErrorTemplateKey, Arc<ErrorTemplate>>>,
}

pub struct JitGadgetState {
    pub gtype: u64,
    pub outputs: Vec<watch::Sender<Option<bin::gadget::Connector>>>,
    /// The expanded output checks for each output port - each check is a set of explicit measurements
    /// paired with the accumulated naturally_flipped parity (affine constant)
    output_checks: Vec<Vec<(HashSet<ExplicitMeasurement>, bool)>>,
    /// build the reverse mapping from input virtual checks to all the finished and unfinished
    /// checks that includes the input virtual check; it has the same type as jit_gadget_type::Error
    /// but doesn't have the base bin::Error information
    pub input_virtual_check_map: Arc<Vec<Vec<jit::jit_gadget_type::Error>>>,
    /// for each gadget, we need to resolve the output virtual checks into concrete checks
    /// indexed by explicit cid and check index
    output_virtual_checks: watch::Sender<Option<Arc<Vec<HashSet<ExplicitCheck>>>>>,
    pub error_model_ready: bool,
    /// GC holds: an unpublished error model, unconnected outputs, and incoming connector/check references.
    /// Releasing the last hold queues this state for collection; this is not an Arc reference count.
    retention_count: usize,
}

/// An explicit measurement with absolute gid instead of remote_gadget index
#[derive(Eq, PartialEq, Hash, Clone, Debug)]
struct ExplicitMeasurement {
    gid: u64,
    measurement_index: u64,
}

#[derive(Eq, PartialEq, Ord, PartialOrd, Hash, Clone, Debug)]
struct ExplicitCheck {
    // in the JIT compiler, cid is always equal to gid for simplicity
    cid: u64,
    check_index: u64,
}

/// Identifies reusable error expansion by gadget type and normalized output-check context.
#[derive(Eq, PartialEq, Hash)]
struct ErrorTemplateKey {
    gtype: u64,
    /// One entry per unfinished output check, not per error.
    /// Each inner vector is its sorted resolved check set, with normalized CIDs and remote check offsets.
    /// Key construction and hashing scale with these references, not the full error list.
    output_checks: Vec<Vec<ExplicitCheck>>,
}

struct ErrorTemplate {
    source_type: std::sync::Weak<jit::JitGadgetType>,
    model_type: Arc<bin::ErrorModelType>,
    remote_bindings: Vec<(usize, u64)>,
}

/// Returns the reusable template key and instance-specific remote sources, sorted by absolute CID.
/// Each source is `(absolute_cid, minimum_check_index)`: normalized CID `i + 1` maps to
/// `sources[i]`, whose minimum index is added back when binding the template to this instance.
/// Normalized CID zero denotes the current gadget and has no entry in the source vector.
fn normalize_error_context(gtype: u64, gid: u64, checks: &[HashSet<ExplicitCheck>]) -> (ErrorTemplateKey, Vec<(u64, u64)>) {
    let mut sources = std::collections::BTreeMap::<u64, u64>::new();
    for check in checks.iter().flatten().filter(|check| check.cid != gid) {
        sources
            .entry(check.cid)
            .and_modify(|minimum| *minimum = (*minimum).min(check.check_index))
            .or_insert(check.check_index);
    }
    let sources: Vec<_> = sources.into_iter().collect();
    let output_checks = checks
        .iter()
        .map(|checks| {
            let mut normalized: Vec<_> = checks
                .iter()
                .map(|check| {
                    if check.cid == gid {
                        ExplicitCheck {
                            cid: 0,
                            check_index: check.check_index,
                        }
                    } else {
                        let index = sources.binary_search_by_key(&check.cid, |&(cid, _)| cid).unwrap();
                        ExplicitCheck {
                            cid: u64::try_from(index).unwrap() + 1,
                            check_index: check.check_index - sources[index].1,
                        }
                    }
                })
                .collect();
            normalized.sort_unstable();
            normalized
        })
        .collect();
    (ErrorTemplateKey { gtype, output_checks }, sources)
}

fn build_error_template(source_type: &Arc<jit::JitGadgetType>, key: &ErrorTemplateKey) -> ErrorTemplate {
    let checks: Vec<HashSet<_>> = key
        .output_checks
        .iter()
        .map(|checks| checks.iter().cloned().collect())
        .collect();
    let mut sequencer = RemoteCheckModelSequencer::new();
    let mut errors = Vec::with_capacity(source_type.errors.len());
    for error in &source_type.errors {
        if error.unfinished_checks.is_empty() {
            errors.push(sequencer.finished_only_to_bin_error(error));
        } else {
            let checks = expand_error_checks(error, &checks, 0);
            errors.push(sequencer.explicit_checks_to_bin_error(error, &checks, 0));
        }
    }
    sequencer.apply_bias_to_errors(&mut errors);
    let remotes = sequencer.finalize();
    let remote_bindings = remotes
        .iter()
        .map(|remote| {
            (
                usize::try_from(remote.absolute_cid.unwrap().checked_sub(1).unwrap()).unwrap(),
                remote.check_bias,
            )
        })
        .collect();
    ErrorTemplate {
        source_type: Arc::downgrade(source_type),
        model_type: Arc::new(bin::ErrorModelType {
            ctype: crate::misc::index::WILDCARD,
            remote_check_models: vec![bin::error_model_type::RemoteCheckModel::default(); remotes.len()],
            errors,
            ..Default::default()
        }),
        remote_bindings,
    }
}

impl JitCompiler {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            jit_port_types: RwLock::new(HashMap::new()),
            jit_gadget_types: RwLock::new(HashMap::new()),
            terminal_error_model_types: RwLock::new(HashMap::new()),
            input_virtual_check_maps: RwLock::new(HashMap::new()),
            gadgets: RwLock::new(HashMap::new()),
            collected_gids: RwLock::default(),
            collection_candidates: std::sync::Mutex::new(Vec::new()),
            error_templates: tokio::sync::Mutex::new(lru::LruCache::new(
                const { std::num::NonZeroUsize::new(128).unwrap() },
            )),
            current_gid: AtomicU64::new(1),
        })
    }

    /// it is unsafe to call reset while some other async operations are ongoing
    ///
    /// # Panics
    /// Panics if an earlier collection operation poisoned the retirement queue.
    pub async fn reset(&self) {
        let mut gadgets = self.gadgets.write().await;
        self.current_gid.store(1, Ordering::SeqCst);
        gadgets.clear();
        self.collection_candidates.lock().unwrap().clear();
        self.collected_gids.write().await.clear();
    }

    /// Clear registered types and compiled gadgets after all ongoing operations have stopped.
    pub async fn reset_library(&self) {
        self.reset().await;
        self.jit_port_types.write().await.clear();
        self.jit_gadget_types.write().await.clear();
        self.terminal_error_model_types.write().await.clear();
        self.input_virtual_check_maps.write().await.clear();
        self.error_templates.lock().await.clear();
    }

    pub async fn contains_gid(&self, gid: u64) -> bool {
        self.gadgets.read().await.contains_key(&gid)
    }

    pub(crate) async fn was_collected(&self, gid: u64) -> bool {
        self.collected_gids.read().await.contains(&gid)
    }

    // Forward port/check references form a DAG. Measurement references are roots,
    // so zero-count retirement is equivalent to tracing without rescanning history.
    pub(crate) async fn collect_garbage(
        &self,
        mut gadgets: tokio::sync::RwLockWriteGuard<'_, HashMap<u64, JitGadgetState>>,
    ) {
        #[cfg(debug_assertions)]
        let expected = Self::traced_live_gadgets(&gadgets);
        let mut collected = self.collected_gids.write().await;
        let mut pending = std::mem::take(&mut *self.collection_candidates.lock().unwrap());
        while let Some(gid) = pending.pop() {
            let Some(gadget) = gadgets.get(&gid) else {
                continue;
            };
            if gadget.retention_count != 0 {
                continue;
            }
            let gadget = gadgets.remove(&gid).unwrap();
            debug_assert!(gadget.error_model_ready);
            debug_assert!(gadget.output_checks.iter().all(Vec::is_empty));
            collected.insert(gid..=gid);
            for peer in &gadget.outputs {
                if let Some(peer) = peer.borrow().as_ref() {
                    Self::release_reference(&mut gadgets, peer.gid, &mut pending);
                }
            }
            if let Some(checks) = gadget.output_virtual_checks.borrow().as_ref() {
                let targets: HashSet<_> = checks.iter().flatten().map(|check| check.cid).collect();
                for target in targets {
                    Self::release_reference(&mut gadgets, target, &mut pending);
                }
            }
        }
        *self.collection_candidates.lock().unwrap() = pending;
        #[cfg(debug_assertions)]
        debug_assert_eq!(gadgets.keys().copied().collect::<HashSet<_>>(), expected);
    }

    fn retain_reference(gadgets: &mut HashMap<u64, JitGadgetState>, gid: u64) {
        let gadget = gadgets.get_mut(&gid).expect("retained JIT dependency must be live");
        gadget.retention_count = gadget.retention_count.checked_add(1).expect("JIT reference count overflow");
    }

    fn release_reference(gadgets: &mut HashMap<u64, JitGadgetState>, gid: u64, pending: &mut Vec<u64>) {
        let gadget = gadgets.get_mut(&gid).expect("released JIT dependency must be live");
        gadget.retention_count = gadget.retention_count.checked_sub(1).expect("JIT reference count underflow");
        if gadget.retention_count == 0 {
            pending.push(gid);
        }
    }

    pub(crate) async fn mark_error_model_ready(&self, gid: u64) {
        let mut gadgets = self.gadgets.write().await;
        let gadget = gadgets
            .get_mut(&gid)
            .expect("unpublished error model must keep its gadget live");
        if !gadget.error_model_ready {
            gadget.error_model_ready = true;
            Self::release_reference(&mut gadgets, gid, &mut self.collection_candidates.lock().unwrap());
        }
    }

    #[cfg(debug_assertions)]
    fn traced_live_gadgets(gadgets: &HashMap<u64, JitGadgetState>) -> HashSet<u64> {
        let mut retained = HashSet::new();
        for (&gid, gadget) in gadgets {
            if !gadget.error_model_ready || gadget.outputs.iter().any(|peer| peer.borrow().is_none()) {
                retained.insert(gid);
            }
            for (measurements, _) in gadget.output_checks.iter().flatten() {
                for measurement in measurements {
                    retained.insert(measurement.gid);
                }
            }
        }
        let mut pending: Vec<_> = retained.iter().copied().collect();
        while let Some(gid) = pending.pop() {
            if let Some(gadget) = gadgets.get(&gid) {
                for peer in &gadget.outputs {
                    if let Some(peer) = peer.borrow().as_ref()
                        && retained.insert(peer.gid)
                    {
                        pending.push(peer.gid);
                    }
                }
                if let Some(checks) = gadget.output_virtual_checks.borrow().as_ref() {
                    for check in checks.iter().flatten() {
                        if retained.insert(check.cid) {
                            pending.push(check.cid);
                        }
                    }
                }
            }
        }
        retained.retain(|gid| gadgets.contains_key(gid));
        retained
    }

    pub(crate) fn assign_gid(&self, gadget: &mut bin::Gadget) {
        if gadget.gid == 0 {
            gadget.gid = self.current_gid.fetch_add(1, Ordering::SeqCst);
        } else {
            self.current_gid.fetch_max(gadget.gid.saturating_add(1), Ordering::SeqCst);
        }
    }

    async fn get_output_connector(
        &self,
        gid: u64,
        port_index: usize,
        token: CancellationToken,
    ) -> Option<bin::gadget::Connector> {
        let mut receiver = {
            let gadgets = self.gadgets.read().await;
            let sender = &gadgets[&gid].outputs[port_index];
            if let Some(value) = *sender.borrow() {
                return Some(value);
            }
            sender.subscribe()
        };
        tokio::select! {
            result = receiver.wait_for(Option::is_some) => result.ok().and_then(|value| *value),
            () = token.cancelled() => None,
        }
    }

    async fn get_input_virtual_check_map(&self, gid: u64) -> Arc<Vec<Vec<jit::jit_gadget_type::Error>>> {
        let gadgets = self.gadgets.read().await;
        gadgets[&gid].input_virtual_check_map.clone()
    }

    async fn get_output_virtual_checks(
        &self,
        gid: u64,
        token: CancellationToken,
    ) -> Option<Arc<Vec<HashSet<ExplicitCheck>>>> {
        let mut receiver = {
            let gadgets = self.gadgets.read().await;
            let sender = &gadgets[&gid].output_virtual_checks;
            if let Some(value) = sender.borrow().as_ref() {
                return Some(Arc::clone(value));
            }
            sender.subscribe()
        };
        tokio::select! {
            result = receiver.wait_for(Option::is_some) => result.ok().and_then(|value| value.clone()),
            () = token.cancelled() => None,
        }
    }

    async fn publish_output_virtual_checks(&self, gid: u64, checks: Arc<Vec<HashSet<ExplicitCheck>>>) {
        let targets: HashSet<_> = checks.iter().flatten().map(|check| check.cid).collect();
        let mut gadgets = self.gadgets.write().await;
        assert!(
            gadgets[&gid].output_virtual_checks.borrow().is_none(),
            "virtual checks were already published"
        );
        for target in targets {
            Self::retain_reference(&mut gadgets, target);
        }
        gadgets[&gid].output_virtual_checks.send_replace(Some(checks));
    }

    pub async fn load_library(&self, library: jit::JitLibrary) {
        let mut jit_port_types = self.jit_port_types.write().await;
        let mut jit_gadget_types = self.jit_gadget_types.write().await;
        let mut terminal_types = self.terminal_error_model_types.write().await;
        let mut input_maps = self.input_virtual_check_maps.write().await;
        for port_type in library.port_types {
            let ptype = port_type.base.as_ref().unwrap().ptype;
            assert!(!jit_port_types.contains_key(&ptype));
            jit_port_types.insert(ptype, Arc::new(port_type));
        }
        for gadget_type in library.gadget_types {
            let gtype = gadget_type.base.as_ref().unwrap().gtype;
            assert!(!jit_gadget_types.contains_key(&gtype));
            if gadget_type
                .base
                .as_ref()
                .unwrap()
                .inputs
                .iter()
                .all(|port| jit_port_types.contains_key(&port.ptype))
            {
                input_maps.insert(gtype, input_virtual_check_map(&gadget_type, &jit_port_types));
            }
            let mut terminal_type = bin::ErrorModelType {
                etype: gtype,
                ..Default::default()
            };
            for error in &gadget_type.errors {
                let mut terminal_error = error.base.clone().unwrap();
                terminal_error.checks.extend(error.finished_checks.iter().map(|&check_index| {
                    bin::error_model_type::RemoteCheck {
                        remote_check_model: None,
                        check_index,
                    }
                }));
                if !error.unfinished_checks.is_empty() {
                    if terminal_type.remote_check_models.is_empty() {
                        terminal_type
                            .remote_check_models
                            .push(bin::error_model_type::RemoteCheckModel {
                                absolute_cid: Some(FUTURE_CHECK_CID),
                                ..Default::default()
                            });
                    }
                    terminal_error.checks.push(bin::error_model_type::RemoteCheck {
                        remote_check_model: Some(0),
                        check_index: 0,
                    });
                }
                terminal_type.errors.push(terminal_error);
            }
            terminal_types.insert(gtype, Arc::new(terminal_type));
            jit_gadget_types.insert(gtype, Arc::new(gadget_type));
        }
        drop(jit_gadget_types);
    }

    /// Compile an instruction and materialize its independently owned error type.
    ///
    /// # Panics
    /// Panics for invalid instructions, inconsistent graph state, or invalid compiled bindings.
    pub async fn compile(
        self: &Arc<Self>,
        instruction: jit::JitInstruction,
        token: CancellationToken,
    ) -> (
        bin::Gadget,
        bin::CheckModelType,
        bin::CheckModel,
        impl Future<Output = (bin::ErrorModelType, bin::ErrorModel)> + Send + use<>,
    ) {
        let (gadget, check_type, check, future) = self.compile_shared(instruction, token).await;
        (gadget, check_type, check, async move {
            let (model_type, mut model) = future.await;
            let mut model_type = model_type.as_ref().clone();
            model_type.etype = model.etype;
            if let Some(modifier) = model.modifier.as_mut() {
                for reroute in modifier.reroute_remote_check_models.drain(..) {
                    let mut remote = reroute.value.expect("compiled remote check binding must be present");
                    // The owned compiler assigns ctype=cid; shared controller types
                    // are deduplicated independently and keep wildcard expectations.
                    remote.expecting_ctype = remote.absolute_cid.expect("compiled binding must have an absolute cid");
                    model_type.remote_check_models[usize::try_from(reroute.remote_check_model_index).unwrap()] = remote;
                }
                if modifier.probability_modifier.is_none() {
                    model.modifier = None;
                }
            }
            (model_type, model)
        })
    }

    #[allow(clippy::too_many_lines, reason = "Keep check preparation and atomic publication adjacent.")]
    pub(crate) async fn compile_shared(
        self: &Arc<Self>,
        mut instruction: jit::JitInstruction,
        token: CancellationToken,
    ) -> (
        bin::Gadget,
        bin::CheckModelType,
        bin::CheckModel,
        // error model comes later asynchronously because it needs to wait for all errors to be
        // expanded into appropriate checks
        impl Future<Output = (Arc<bin::ErrorModelType>, bin::ErrorModel)> + Send + use<>,
    ) {
        let jit_port_types = self.jit_port_types.read().await;
        let jit_gadget_types = self.jit_gadget_types.read().await;
        let mut gadget = instruction.gadget.take().unwrap();
        let gid = if gadget.gid != 0 {
            // User-provided gid: advance current_gid past it to avoid conflicts
            self.current_gid.fetch_max(gadget.gid + 1, Ordering::SeqCst);
            gadget.gid
        } else {
            let gid = self.current_gid.fetch_add(1, Ordering::SeqCst);
            gadget.gid = gid;
            gid
        };
        let probability_modifier = instruction.probability_modifier.take();
        let gtype = gadget.gtype;
        let jit_gadget_type = jit_gadget_types
            .get(&gtype)
            .unwrap_or_else(|| panic!("Gadget type not found: {}", gtype))
            .clone();
        let gadget_type = jit_gadget_type.base.as_ref().unwrap();
        // Collect the output checks from each input port's input gadget
        let mut input_checks: Vec<Vec<(HashSet<ExplicitMeasurement>, bool)>> = vec![];
        let cached_input_map = self.input_virtual_check_maps.read().await.get(&gtype).cloned();
        let input_virtual_check_map = if let Some(map) = cached_input_map {
            map
        } else {
            let map = input_virtual_check_map(&jit_gadget_type, &jit_port_types);
            self.input_virtual_check_maps.write().await.insert(gtype, Arc::clone(&map));
            map
        };
        let gadgets = self.gadgets.read().await;
        for (input_port_index, connector) in gadget.connectors.iter().enumerate() {
            let output_port = usize::try_from(connector.port).expect("output port exceeds usize");
            assert!(
                !gadget.connectors[..input_port_index].contains(connector),
                "input port is connected more than once"
            );
            debug_assert!(gadgets.contains_key(&connector.gid));
            debug_assert!({
                let peer_outputs = &gadgets[&connector.gid].outputs;
                output_port < peer_outputs.len() && peer_outputs[output_port].borrow().is_none()
            });
            let input_gadget = &gadgets[&connector.gid];
            // Keep the input checks published until the new gadget is committed atomically.
            let peer_output_checks = input_gadget.output_checks[output_port].clone();
            debug_assert_eq!(input_virtual_check_map[input_port_index].len(), peer_output_checks.len());
            input_checks.push(peer_output_checks);
        }
        // expand internal checks and add them to the generated check model
        let mut remote_gadget_sequencer = RemoteGadgetSequencer::new();
        let mut check_model_type = bin::CheckModelType {
            ctype: gid, // need to create a new check model every time
            gtype,
            ..Default::default()
        };
        for check in &jit_gadget_type.finished_checks {
            let (measurements, naturally_flipped) = expand_check_measurements(check, &input_checks, gid);
            check_model_type
                .checks
                .push(remote_gadget_sequencer.measurements_to_bin_check(
                    check,
                    &measurements,
                    naturally_flipped,
                    gid,
                    &gadgets,
                ));
        }
        // Build the output_checks for this gadget - each port has a list of expanded measurement sets
        let mut output_checks = vec![];
        let mut output_check_index = 0;
        for port in gadget_type.outputs.iter() {
            let port_type = jit_port_types.get(&port.ptype).unwrap();
            let mut port_checks = vec![];
            for _ in 0..port_type.stabilizers.len() {
                let check = &jit_gadget_type.unfinished_checks[output_check_index];
                let expanded = expand_check_measurements(check, &input_checks, gid);
                port_checks.push(expanded);
                output_check_index += 1;
            }
            output_checks.push(port_checks);
        }
        debug_assert!(output_check_index == jit_gadget_type.unfinished_checks.len());
        // Finalize checks by subtracting measurement bias from remote measurements
        remote_gadget_sequencer.apply_bias_to_check_model(&mut check_model_type);
        // Assign the sequenced remote_gadgets to the check_model_type
        check_model_type.remote_gadgets = remote_gadget_sequencer.finalize();
        let gadget_state = JitGadgetState {
            error_model_ready: false,
            retention_count: 1 + gadget_type.outputs.len(),
            gtype,
            outputs: gadget_type.outputs.iter().map(|_| watch::channel(None).0).collect(),
            output_checks,
            input_virtual_check_map,
            output_virtual_checks: watch::channel(None).0,
        };
        let check_model = bin::CheckModel {
            gid,
            ctype: gid, // need to create a new check model every time
            cid: gid,
            terminal_error_model: Some(bin::ErrorModel {
                etype: gtype,
                cid: gid,
                eid: gid,
                modifier: probability_modifier
                    .clone()
                    .map(|modifier| bin::error_model::ErrorModelModifier {
                        probability_modifier: Some(modifier),
                        ..Default::default()
                    }),
                ..Default::default()
            }),
            ..Default::default()
        };
        drop(jit_port_types);
        drop(jit_gadget_types);
        drop(gadgets);
        {
            let mut gadgets = self.gadgets.write().await;
            assert!(!gadgets.contains_key(&gid), "gadget id was concurrently registered");
            for connector in &gadget.connectors {
                let output_port = usize::try_from(connector.port).expect("output port exceeds usize");
                assert!(
                    gadgets[&connector.gid].outputs[output_port].borrow().is_none(),
                    "input port was concurrently connected"
                );
            }
            let measurement_roots: Vec<_> = gadget_state
                .output_checks
                .iter()
                .flatten()
                .flat_map(|(measurements, _)| measurements.iter().map(|measurement| measurement.gid))
                .collect();
            gadgets.insert(gid, gadget_state);
            for root in measurement_roots {
                Self::retain_reference(&mut gadgets, root);
            }
            let mut pending = self.collection_candidates.lock().unwrap();
            for (input_port_index, connector) in gadget.connectors.iter().enumerate() {
                let output_port = usize::try_from(connector.port).expect("output port exceeds usize");
                let input = gadgets.get_mut(&connector.gid).unwrap();
                let consumed = std::mem::take(&mut input.output_checks[output_port]);
                input.outputs[output_port].send_replace(Some(bin::gadget::Connector {
                    gid,
                    port: input_port_index as u64,
                }));
                Self::retain_reference(&mut gadgets, gid);
                Self::release_reference(&mut gadgets, connector.gid, &mut pending);
                for (measurements, _) in consumed {
                    for measurement in measurements {
                        Self::release_reference(&mut gadgets, measurement.gid, &mut pending);
                    }
                }
            }
        }
        let this = Arc::clone(self);
        (gadget, check_model_type, check_model, async move {
            // to avoid infinite dependency chain to the final measurement gadget, we need to achieve this property:
            // if a gadget's unfinished checks do not ever refer to the gadget's input checks, then it can immediately
            // resolve its first-stage check state
            let mut output_virtual_checks = Vec::with_capacity(jit_gadget_type.unfinished_checks.len());
            let mut cancelled = false;
            for port_index in 0..jit_gadget_type.base.as_ref().unwrap().outputs.len() {
                // wait for the output port to be connected; otherwise we won't be able to resolve the checks
                let connector = this.get_output_connector(gid, port_index, token.clone()).await;
                let Some(connector) = connector else {
                    cancelled = true;
                    break;
                };
                let peer_input_check_map = this.get_input_virtual_check_map(connector.gid).await;
                let peer_input_checks = &peer_input_check_map[connector.port as usize];
                // iterate over all the output virtual checks on this port and resolve them using
                // the resolved information from the peer gadget
                let mut peer_output_virtual_checks: Option<Arc<Vec<HashSet<ExplicitCheck>>>> = None;
                for check in peer_input_checks.iter() {
                    // expanding the virtual check into explicit checks
                    let mut explicit_checks = HashSet::<ExplicitCheck>::new();
                    for &check_index in check.finished_checks.iter() {
                        let explicit_check = ExplicitCheck {
                            cid: connector.gid,
                            check_index,
                        };
                        debug_assert!(!explicit_checks.contains(&explicit_check));
                        explicit_checks.insert(explicit_check);
                    }
                    // if there exists any unfinished checks, we must wait for the peer's output virtual checks
                    // to be resolved first (it is done in the peer's own error resolving Future function)
                    // note that this is rarely happening
                    if !check.unfinished_checks.is_empty() {
                        if peer_output_virtual_checks.is_none() {
                            peer_output_virtual_checks = this.get_output_virtual_checks(connector.gid, token.clone()).await;
                        }
                        let Some(peer_output_virtual_checks_arc) = peer_output_virtual_checks.as_ref() else {
                            cancelled = true;
                            break;
                        };
                        for &check_index in check.unfinished_checks.iter() {
                            let peer_explicit_checks = &peer_output_virtual_checks_arc[check_index as usize];
                            for check in peer_explicit_checks {
                                if !explicit_checks.remove(check) {
                                    explicit_checks.insert(check.clone());
                                }
                            }
                        }
                    }
                    output_virtual_checks.push(explicit_checks);
                }
                if cancelled {
                    break;
                }
            }
            // If cancelled, return a dummy error model. The caller (JitController)
            // wraps this future in select! against the token, so the result is
            // discarded. We must NOT publish partial output_virtual_checks because
            // other in-flight futures could read them and panic on bad indices.
            if cancelled {
                return (Arc::new(bin::ErrorModelType::default()), bin::ErrorModel::default());
            }
            let output_virtual_checks = Arc::new(output_virtual_checks);
            debug_assert!(output_virtual_checks.len() == jit_gadget_type.unfinished_checks.len());
            // once the output virtual checks are resolved, we can update the state so that
            // the previous gadgets depending on this gadget can proceed
            this.publish_output_virtual_checks(gid, output_virtual_checks.clone()).await;
            let (key, sources) = normalize_error_context(gtype, gid, &output_virtual_checks);
            let mut cache = this.error_templates.lock().await;
            let template = if let Some(template) = cache
                .get(&key)
                .filter(|template| std::ptr::eq(template.source_type.as_ptr(), Arc::as_ptr(&jit_gadget_type)))
            {
                Arc::clone(template)
            } else {
                let template = Arc::new(build_error_template(&jit_gadget_type, &key));
                cache.put(key, Arc::clone(&template));
                template
            };
            drop(cache);
            let reroutes: Vec<_> = template
                .remote_bindings
                .iter()
                .enumerate()
                .map(
                    |(index, &(source, bias))| bin::error_model::error_model_modifier::RerouteRemoteCheckModel {
                        remote_check_model_index: u64::try_from(index).unwrap(),
                        value: Some(bin::error_model_type::RemoteCheckModel {
                            absolute_cid: Some(sources[source].0),
                            check_bias: sources[source].1.checked_add(bias).expect("remote check bias overflow"),
                            ..Default::default()
                        }),
                    },
                )
                .collect();
            let error_model = bin::ErrorModel {
                cid: gid,
                etype: gid,
                eid: gid,
                modifier: (probability_modifier.is_some() || !reroutes.is_empty()).then_some(
                    bin::error_model::ErrorModelModifier {
                        probability_modifier,
                        reroute_remote_check_models: reroutes,
                    },
                ),
                ..Default::default()
            };
            (Arc::clone(&template.model_type), error_model)
        })
    }
}

/// Helper struct to sequence remote gadgets across multiple checks
struct RemoteGadgetSequencer {
    gid_to_remote_index: HashMap<u64, u64>,
    remote_gadgets: Vec<bin::check_model_type::RemoteGadget>,
    minimum_measurement_indices: Vec<u64>,
}

impl RemoteGadgetSequencer {
    fn new() -> Self {
        Self {
            gid_to_remote_index: HashMap::new(),
            remote_gadgets: vec![],
            minimum_measurement_indices: vec![],
        }
    }

    fn get_or_insert(&mut self, gid: u64, gtype: u64, measurement_index: u64) -> u64 {
        if let Some(&index) = self.gid_to_remote_index.get(&gid) {
            self.minimum_measurement_indices[index as usize] =
                std::cmp::min(self.minimum_measurement_indices[index as usize], measurement_index);
            index
        } else {
            let index = self.remote_gadgets.len() as u64;
            self.gid_to_remote_index.insert(gid, index);
            self.remote_gadgets.push(bin::check_model_type::RemoteGadget {
                absolute_gid: Some(gid),
                expecting_gtype: gtype,
                ..Default::default()
            });
            self.minimum_measurement_indices.push(measurement_index);
            index
        }
    }

    fn get_measurement_bias(&self, remote_index: u64) -> u64 {
        self.minimum_measurement_indices[remote_index as usize]
    }

    /// Convert a set of explicit measurements to a bin::check_model_type::Check
    fn measurements_to_bin_check(
        &mut self,
        jit_check: &jit::jit_gadget_type::Check,
        measurements: &HashSet<ExplicitMeasurement>,
        naturally_flipped: bool,
        current_gid: u64,
        gadgets: &HashMap<u64, JitGadgetState>,
    ) -> bin::check_model_type::Check {
        let mut check = jit_check.base.as_ref().unwrap().clone();
        check.naturally_flipped = naturally_flipped;
        debug_assert!(check.measurements.is_empty());
        for measurement in measurements.iter() {
            let (remote_gadget, measurement_index) = if measurement.gid != current_gid {
                let remote_gtype = gadgets[&measurement.gid].gtype;
                let remote_index = self.get_or_insert(measurement.gid, remote_gtype, measurement.measurement_index);
                (Some(remote_index), measurement.measurement_index)
            } else {
                (None, measurement.measurement_index)
            };
            check.measurements.push(bin::check_model_type::RemoteMeasurement {
                remote_gadget,
                measurement_index,
            });
        }
        // Sort measurements for deterministic ordering (HashSet iteration is
        // non-deterministic). This is critical for type cache hits across shots.
        check.measurements.sort_by(|a, b| {
            a.remote_gadget
                .cmp(&b.remote_gadget)
                .then(a.measurement_index.cmp(&b.measurement_index))
        });
        check
    }

    /// Finalize checks by subtracting measurement bias from remote measurements
    fn apply_bias_to_check_model(&self, check_model_type: &mut bin::CheckModelType) {
        for check in check_model_type.checks.iter_mut() {
            for measurement in check.measurements.iter_mut() {
                if let Some(remote_index) = measurement.remote_gadget {
                    measurement.measurement_index -= self.get_measurement_bias(remote_index);
                }
            }
        }
    }

    fn finalize(mut self) -> Vec<bin::check_model_type::RemoteGadget> {
        for (index, remote_gadget) in self.remote_gadgets.iter_mut().enumerate() {
            remote_gadget.measurement_bias = self.minimum_measurement_indices[index];
        }
        self.remote_gadgets
    }
}

/// Helper struct to sequence remote check models across multiple errors
struct RemoteCheckModelSequencer {
    cid_to_remote_index: HashMap<u64, u64>,
    remote_check_models: Vec<bin::error_model_type::RemoteCheckModel>,
    minimum_check_indices: Vec<u64>,
}

impl RemoteCheckModelSequencer {
    fn new() -> Self {
        Self {
            cid_to_remote_index: HashMap::new(),
            remote_check_models: vec![],
            minimum_check_indices: vec![],
        }
    }

    fn get_or_insert(&mut self, cid: u64, check_index: u64) -> u64 {
        if let Some(&index) = self.cid_to_remote_index.get(&cid) {
            self.minimum_check_indices[index as usize] =
                std::cmp::min(self.minimum_check_indices[index as usize], check_index);
            index
        } else {
            let index = self.remote_check_models.len() as u64;
            self.cid_to_remote_index.insert(cid, index);
            self.remote_check_models.push(bin::error_model_type::RemoteCheckModel {
                expecting_ctype: cid,
                absolute_cid: Some(cid),
                ..Default::default()
            });
            self.minimum_check_indices.push(check_index);
            index
        }
    }

    fn get_check_bias(&self, remote_index: u64) -> u64 {
        self.minimum_check_indices[remote_index as usize]
    }

    /// Fast path for errors that only reference finished checks (no unfinished checks).
    /// All finished checks are local (cid = current_gid, remote_check_model = None),
    /// so we skip HashSet construction, hashing, and sorting entirely.
    fn finished_only_to_bin_error(&mut self, jit_error: &jit::jit_gadget_type::Error) -> bin::error_model_type::Error {
        debug_assert!(jit_error.unfinished_checks.is_empty());
        let mut error = jit_error.base.clone().unwrap();
        // Finished checks are already sorted by index in the JIT representation,
        // and all are local (no remote_check_model), so no sorting needed.
        for &check_index in jit_error.finished_checks.iter() {
            error.checks.push(bin::error_model_type::RemoteCheck {
                remote_check_model: None,
                check_index,
            });
        }
        error
    }

    /// Convert a set of explicit checks to a bin::error_model_type::Error
    fn explicit_checks_to_bin_error(
        &mut self,
        jit_error: &jit::jit_gadget_type::Error,
        explicit_checks: &HashSet<ExplicitCheck>,
        current_gid: u64,
    ) -> bin::error_model_type::Error {
        let mut error = jit_error.base.clone().unwrap();
        for explicit_check in explicit_checks.iter() {
            let cid = explicit_check.cid;
            let remote_check_model = if cid == current_gid {
                None
            } else {
                Some(self.get_or_insert(cid, explicit_check.check_index))
            };
            error.checks.push(bin::error_model_type::RemoteCheck {
                remote_check_model,
                check_index: explicit_check.check_index,
            });
        }
        // Sort checks for deterministic ordering (HashSet iteration is
        // non-deterministic). This is critical for type cache hits across shots.
        error.checks.sort_by(|a, b| {
            a.remote_check_model
                .cmp(&b.remote_check_model)
                .then(a.check_index.cmp(&b.check_index))
        });
        error
    }

    /// Apply bias to all errors by subtracting the minimum check index
    fn apply_bias_to_errors(&self, errors: &mut [bin::error_model_type::Error]) {
        for error in errors.iter_mut() {
            for remote_check in error.checks.iter_mut() {
                remote_check.check_index -= match remote_check.remote_check_model {
                    Some(remote_index) => self.get_check_bias(remote_index),
                    None => 0,
                };
            }
        }
    }

    fn finalize(mut self) -> Vec<bin::error_model_type::RemoteCheckModel> {
        for (index, remote_check_model) in self.remote_check_models.iter_mut().enumerate() {
            remote_check_model.check_bias = self.minimum_check_indices[index];
        }
        self.remote_check_models
    }
}

/// Expand a JIT check into a set of explicit measurements by expanding references to input port measurements.
/// Returns the measurement set and the accumulated `naturally_flipped` parity (affine constant).
fn expand_check_measurements(
    check: &jit::jit_gadget_type::Check,
    input_checks: &[Vec<(HashSet<ExplicitMeasurement>, bool)>],
    gid: u64,
) -> (HashSet<ExplicitMeasurement>, bool) {
    let mut naturally_flipped = check.base.as_ref().unwrap().naturally_flipped;
    let mut measurement_set = HashSet::<ExplicitMeasurement>::new();
    for measurement in check.measurements.iter() {
        if let Some(input_port) = measurement.input_port {
            let input_port_idx = input_port as usize;
            let measurement_idx = measurement.measurement_index as usize;
            assert!(
                input_port_idx < input_checks.len() && measurement_idx < input_checks[input_port_idx].len(),
                "malformed JitLibrary: check measurement references input port {input_port} \
                 measurement {measurement_idx}, but input port has {} expanded checks \
                 (this typically means a COMPOSE bound the same wire to multiple INPUT or \
                 OUTPUT ports, or that two connectors share the same source (gid, port))",
                input_checks.get(input_port_idx).map_or(0, std::vec::Vec::len),
            );
            let (input_measurements, input_flipped) = &input_checks[input_port_idx][measurement_idx];
            naturally_flipped ^= input_flipped;
            for input_measurement in input_measurements.iter() {
                if !measurement_set.remove(input_measurement) {
                    measurement_set.insert(input_measurement.clone());
                }
            }
        } else {
            let explicit_measurement = ExplicitMeasurement {
                gid,
                measurement_index: measurement.measurement_index,
            };
            if !measurement_set.remove(&explicit_measurement) {
                measurement_set.insert(explicit_measurement);
            }
        }
    }
    (measurement_set, naturally_flipped)
}

/// Expand a JIT error into a set of explicit checks by expanding references to unfinished checks
fn expand_error_checks(
    jit_error: &jit::jit_gadget_type::Error,
    output_virtual_checks: &[HashSet<ExplicitCheck>],
    gid: u64,
) -> HashSet<ExplicitCheck> {
    let mut check_set = HashSet::<ExplicitCheck>::new();
    // Start with finished checks (local to this gadget)
    for &check_index in jit_error.finished_checks.iter() {
        let explicit_check = ExplicitCheck { cid: gid, check_index };
        if !check_set.remove(&explicit_check) {
            check_set.insert(explicit_check);
        }
    }
    // XOR (symmetric difference) with each unfinished check's resolved checks
    for &check_index in jit_error.unfinished_checks.iter() {
        let resolved_checks = &output_virtual_checks[check_index as usize];
        for resolved_check in resolved_checks.iter() {
            if !check_set.remove(resolved_check) {
                check_set.insert(resolved_check.clone());
            }
        }
    }
    check_set
}

#[cfg(test)]
#[path = "../../tests/unit/jit_compiler_test.rs"]
mod error_template_tests;
