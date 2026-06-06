use crate::misc::sync::get_or_receiver;
use crate::{bin, jit};
use hashbrown::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::{RwLock, watch};
use tokio_util::sync::CancellationToken;

pub struct JitCompiler {
    pub jit_port_types: RwLock<HashMap<u64, Arc<jit::JitPortType>>>,
    pub jit_gadget_types: RwLock<HashMap<u64, Arc<jit::JitGadgetType>>>,
    pub current_gid: AtomicU64,
    pub gadgets: RwLock<HashMap<u64, JitGadgetState>>,
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
}

/// An explicit measurement with absolute gid instead of remote_gadget index
#[derive(Eq, PartialEq, Hash, Clone, Debug)]
struct ExplicitMeasurement {
    gid: u64,
    measurement_index: u64,
}

#[derive(Eq, PartialEq, Hash, Clone, Debug)]
struct ExplicitCheck {
    // in the JIT compiler, cid is always equal to gid for simplicity
    cid: u64,
    check_index: u64,
}

impl JitCompiler {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            jit_port_types: RwLock::new(HashMap::new()),
            jit_gadget_types: RwLock::new(HashMap::new()),
            gadgets: RwLock::new(HashMap::new()),
            current_gid: AtomicU64::new(1),
        })
    }

    /// it is unsafe to call reset while some other async operations are ongoing
    pub async fn reset(&self) {
        let mut gadgets = self.gadgets.write().await;
        self.current_gid.store(1, Ordering::SeqCst);
        gadgets.clear();
    }

    pub async fn contains_gid(&self, gid: u64) -> bool {
        self.gadgets.read().await.contains_key(&gid)
    }

    async fn get_output_connector(
        &self,
        gid: u64,
        port_index: usize,
        token: CancellationToken,
    ) -> Option<bin::gadget::Connector> {
        let gadgets = self.gadgets.read().await;
        let connector = get_or_receiver(&gadgets[&gid].outputs[port_index], token);
        drop(gadgets);
        match connector {
            Ok(value) => Some(value),
            Err(handle) => handle.await.unwrap_or(None),
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
        let gadgets = self.gadgets.read().await;
        let receiver = get_or_receiver(&gadgets[&gid].output_virtual_checks, token);
        drop(gadgets);
        match receiver {
            Ok(value) => Some(value),
            Err(handle) => handle.await.unwrap_or(None),
        }
    }

    async fn publish_output_virtual_checks(&self, gid: u64, checks: Arc<Vec<HashSet<ExplicitCheck>>>) {
        let gadgets = self.gadgets.read().await;
        gadgets[&gid].output_virtual_checks.send_replace(Some(checks));
    }

    pub async fn load_library(&self, library: jit::JitLibrary) {
        let mut jit_port_types = self.jit_port_types.write().await;
        let mut jit_gadget_types = self.jit_gadget_types.write().await;
        for port_type in library.port_types {
            let ptype = port_type.base.as_ref().unwrap().ptype;
            assert!(!jit_port_types.contains_key(&ptype));
            jit_port_types.insert(ptype, Arc::new(port_type));
        }
        drop(jit_port_types);
        for gadget_type in library.gadget_types {
            let gtype = gadget_type.base.as_ref().unwrap().gtype;
            assert!(!jit_gadget_types.contains_key(&gtype));
            jit_gadget_types.insert(gtype, Arc::new(gadget_type));
        }
        drop(jit_gadget_types);
    }

    pub async fn compile(
        self: &Arc<Self>,
        mut instruction: jit::JitInstruction,
        token: CancellationToken,
    ) -> (
        bin::Gadget,
        bin::CheckModelType,
        bin::CheckModel,
        // error model comes later asynchronously because it needs to wait for all errors to be
        // expanded into appropriate checks
        impl Future<Output = (bin::ErrorModelType, bin::ErrorModel)> + Send + use<>,
    ) {
        let jit_port_types = self.jit_port_types.read().await;
        let jit_gadget_types = self.jit_gadget_types.read().await;
        let mut gadgets = self.gadgets.write().await;
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
        let mut input_virtual_check_map = vec![vec![]; gadget.connectors.len()];
        for (input_port_index, connector) in gadget.connectors.iter().enumerate() {
            debug_assert!(gadgets.contains_key(&connector.gid));
            debug_assert!({
                let peer_outputs = &gadgets[&connector.gid].outputs;
                (connector.port as usize) < peer_outputs.len() && peer_outputs[connector.port as usize].borrow().is_none()
            });
            let input_gadget = gadgets.get_mut(&connector.gid).unwrap();
            input_gadget.outputs[connector.port as usize].send_replace(Some(bin::gadget::Connector {
                gid,
                port: input_port_index as u64,
            }));
            // Take the output checks from this port of the input gadget
            let peer_output_checks = std::mem::take(&mut input_gadget.output_checks[connector.port as usize]);
            input_virtual_check_map[input_port_index] =
                vec![jit::jit_gadget_type::Error::default(); peer_output_checks.len()];
            input_checks.push(peer_output_checks);
        }
        // expand internal checks and add them to the generated check model
        let mut remote_gadget_sequencer = RemoteGadgetSequencer::new();
        let mut check_model_type = bin::CheckModelType {
            ctype: gid, // need to create a new check model every time
            gtype,
            ..Default::default()
        };
        for (finished_check_index, check) in jit_gadget_type.finished_checks.iter().enumerate() {
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
            for measurement in check.measurements.iter() {
                if let Some(input_port) = measurement.input_port {
                    input_virtual_check_map[input_port as usize][measurement.measurement_index as usize]
                        .finished_checks
                        .push(finished_check_index as u64);
                }
            }
        }
        for (unfinished_check_index, check) in jit_gadget_type.unfinished_checks.iter().enumerate() {
            for measurement in check.measurements.iter() {
                if let Some(input_port) = measurement.input_port {
                    input_virtual_check_map[input_port as usize][measurement.measurement_index as usize]
                        .unfinished_checks
                        .push(unfinished_check_index as u64);
                }
            }
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
            gtype,
            outputs: gadget_type.outputs.iter().map(|_| watch::channel(None).0).collect(),
            output_checks,
            input_virtual_check_map: Arc::new(input_virtual_check_map),
            output_virtual_checks: watch::channel(None).0,
        };
        gadgets.insert(gid, gadget_state);
        let check_model = bin::CheckModel {
            gid,
            ctype: gid, // need to create a new check model every time
            cid: gid,
            ..Default::default()
        };
        drop(jit_port_types);
        drop(jit_gadget_types);
        drop(gadgets);
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
                            explicit_checks = explicit_checks.symmetric_difference(peer_explicit_checks).cloned().collect();
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
                return (bin::ErrorModelType::default(), bin::ErrorModel::default());
            }
            let output_virtual_checks = Arc::new(output_virtual_checks);
            debug_assert!(output_virtual_checks.len() == jit_gadget_type.unfinished_checks.len());
            // once the output virtual checks are resolved, we can update the state so that
            // the previous gadgets depending on this gadget can proceed
            this.publish_output_virtual_checks(gid, output_virtual_checks.clone()).await;
            // after we have prepared the data for the previous gadgets to consume, we then focus on myself generating
            // errors and also the remote check model entries as necessary; we build the remote check models in a way
            // that maximizes the probability of error model reusing: we iterate over errors and for each check that it
            // refers to, we build the remote check model entry. This means that regardless of how many hops it takes to
            // reach the target check, the error's relative check indices remain unchanged, thus they can be reused easily
            let mut remote_check_model_sequencer = RemoteCheckModelSequencer::new();
            // For each error, expand unfinished checks using symmetric difference (XOR)
            let mut errors = Vec::with_capacity(jit_gadget_type.errors.len());
            for jit_error in jit_gadget_type.errors.iter() {
                if jit_error.unfinished_checks.is_empty() {
                    // Fast path: only finished checks — all local, no HashSet needed
                    errors.push(remote_check_model_sequencer.finished_only_to_bin_error(jit_error));
                } else {
                    let explicit_checks = expand_error_checks(jit_error, &output_virtual_checks, gid);
                    errors.push(remote_check_model_sequencer.explicit_checks_to_bin_error(jit_error, &explicit_checks, gid));
                }
            }
            // Apply bias to all errors
            remote_check_model_sequencer.apply_bias_to_errors(&mut errors);
            let remote_check_models = remote_check_model_sequencer.finalize();
            // now construct the error model
            let error_model_type = bin::ErrorModelType {
                etype: gid, // need to create a new error model every time
                ctype: crate::misc::index::WILDCARD,
                remote_check_models,
                errors,
                ..Default::default()
            };
            let error_model = bin::ErrorModel {
                cid: gid,
                etype: gid,
                eid: gid,
                modifier: probability_modifier.map(|pm| bin::error_model::ErrorModelModifier {
                    probability_modifier: Some(pm),
                    reroute_remote_check_models: vec![],
                }),
                ..Default::default()
            };
            (error_model_type, error_model)
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
            let (input_measurements, input_flipped) =
                &input_checks[input_port as usize][measurement.measurement_index as usize];
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
