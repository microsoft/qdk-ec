use super::frame::FrameType;
use crate::bin;
use crate::coordinator::reweight_handler::{deduplicate_decoder_input, edge_isolated_vertices};
use crate::decoder::SharedDecodingHypergraph;
use crate::decoder::blackbox_decoder::{DecodingHypergraph, Hyperedge};
use crate::misc::index::ErrorIndex;
use crate::misc::validation::{apply_check_model_reroutes, apply_error_model_reroutes};
use hashbrown::HashMap;
use prost::Message;
use std::sync::{Arc, OnceLock};

pub(super) const SYNDROME: usize = 0;
pub(super) const WINDOW: usize = 1;
pub(super) const FRAME: usize = 2;

pub(super) struct CheckType {
    pub id: u64,
    pub rows: Vec<(bool, Vec<(usize, usize)>)>,
    source_limits: Vec<(usize, usize)>,
}

struct ErrorLimits {
    residual_bits: usize,
    readout_bits: usize,
    checks: Vec<(Option<usize>, usize)>,
}

pub(super) struct CheckPlan {
    pub kind: Arc<CheckType>,
    pub bases: Vec<usize>,
    pub last_input: usize,
}

pub(super) struct FrameInput {
    pub gid: usize,
    pub offset: usize,
    pub bits: usize,
}

pub(super) struct GadgetPlan {
    pub frame: Arc<FrameType>,
    pub inputs: Vec<FrameInput>,
    pub remote_readouts: Vec<(usize, usize)>,
    pub measurement_start: usize,
    pub measurements: usize,
}

pub(super) struct ErrorPlan {
    pub kind: Arc<bin::ErrorModelType>,
    pub remotes: Vec<Option<(usize, usize)>>,
    pub support: Vec<usize>,
}

#[derive(Clone, Hash, PartialEq, Eq)]
struct ErrorPattern {
    etype: u64,
    local: Option<usize>,
    remotes: Vec<Option<(usize, usize)>>,
    owned: bool,
}

#[derive(Hash, PartialEq, Eq)]
struct WindowKey {
    checks: Vec<u64>,
    errors: Vec<ErrorPattern>,
}

pub(super) struct Template {
    pub graph: Arc<SharedDecodingHypergraph>,
    pub edge_errors: Vec<Option<usize>>,
    pub isolated: Vec<usize>,
    pub starts: Vec<usize>,
    pub hid: OnceLock<u64>,
}

pub(super) struct WindowPlan {
    pub template: Arc<Template>,
    pub checks: Vec<usize>,
    pub predecessors: Vec<usize>,
    pub last_input: usize,
}

pub(super) struct Adjacency {
    offsets: Vec<usize>,
    values: Vec<usize>,
}

impl Adjacency {
    fn build(count: usize, edges: &[(usize, usize)]) -> Self {
        let mut offsets = vec![0; count + 1];
        for &(source, _) in edges {
            offsets[source + 1] += 1;
        }
        for index in 1..offsets.len() {
            offsets[index] += offsets[index - 1];
        }
        let mut positions = offsets.clone();
        let mut values = vec![0; edges.len()];
        for &(source, target) in edges {
            values[positions[source]] = target;
            positions[source] += 1;
        }
        Self { offsets, values }
    }

    pub fn get(&self, source: usize) -> &[usize] {
        &self.values[self.offsets[source]..self.offsets[source + 1]]
    }
}

pub(super) struct Plan {
    pub gadgets: Vec<GadgetPlan>,
    pub checks: Vec<CheckPlan>,
    pub errors: Vec<ErrorPlan>,
    pub windows: Vec<Option<WindowPlan>>,
    pub templates: Vec<Arc<Template>>,
    pub dependents: Adjacency,
    pub input_dependents: Adjacency,
    pub remaining: Vec<u32>,
    pub total_measurements: usize,
}

fn index(id: u64, length: usize) -> Result<usize, String> {
    let id = usize::try_from(id).map_err(|_| "ID exceeds usize")?;
    if id == 0 || id > length {
        return Err(format!("ID {id} is outside the compiled program"));
    }
    Ok(id - 1)
}

fn type_table<T>(values: Vec<T>, key: impl Fn(&T) -> u64) -> Result<HashMap<u64, T>, String> {
    let mut table = HashMap::new();
    for value in values {
        let id = key(&value);
        if table.insert(id, value).is_some() {
            return Err(format!("duplicate compiled type ID {id}"));
        }
    }
    Ok(table)
}

fn hop(
    gid: Option<usize>,
    port: Option<(bool, u64)>,
    gadgets: &[bin::Gadget],
    outputs: &[Vec<Option<usize>>],
) -> Result<Option<usize>, String> {
    let Some(gid) = gid else { return Ok(None) };
    let (output, port) = port.ok_or("remote reference needs a port or absolute ID")?;
    let port = usize::try_from(port).map_err(|_| "port exceeds usize")?;
    if output {
        outputs[gid]
            .get(port)
            .copied()
            .ok_or_else(|| "remote output port is out of range".into())
    } else {
        let input = gadgets[gid].connectors.get(port).ok_or("remote input port is out of range")?;
        Ok(Some(index(input.gid, gadgets.len())?))
    }
}

impl Plan {
    #[allow(
        clippy::too_many_lines,
        reason = "This untimed compilation pass validates and binds the complete finite graph before publishing any plan."
    )]
    pub fn compile(library: bin::Library, max_templates: usize) -> Result<Self, String> {
        let count = library
            .program
            .iter()
            .filter(|item| matches!(item.create, Some(bin::instruction::Create::Gadget(_))))
            .count();
        if count == 0 || count > u32::MAX as usize / 3 {
            return Err("compiled-window requires a nonempty, bounded dense-ID program".into());
        }
        let port_types = type_table(library.port_types, |kind| kind.ptype)?;
        let gadget_types = type_table(library.gadget_types, |kind| kind.gtype)?;
        let check_types = type_table(library.check_model_types, |kind| kind.ctype)?;
        let error_types = type_table(library.error_model_types.into_iter().map(Arc::new).collect(), |kind| {
            kind.etype
        })?;
        if gadget_types.values().any(|kind| kind.loss_model.is_some()) {
            return Err("compiled-window does not support loss models".into());
        }
        for kind in error_types.values() {
            if kind
                .errors
                .iter()
                .any(|error| !error.probability.is_finite() || !(0.0..0.5).contains(&error.probability))
            {
                return Err("compiled-window requires fixed error probabilities in [0, 0.5)".into());
            }
        }
        let mut source_gadgets = vec![bin::Gadget::default(); count];
        let mut source_checks = vec![bin::CheckModel::default(); count];
        let mut source_errors = vec![bin::ErrorModel::default(); count];
        let mut next_gadget = 0;
        for instruction in library.program {
            match instruction.create.ok_or("empty program instruction")? {
                bin::instruction::Create::Gadget(value) => {
                    let position = index(value.gid, count)?;
                    if position != next_gadget {
                        return Err("gadget instructions must have ascending dense IDs".into());
                    }
                    next_gadget += 1;
                    if source_gadgets[position].gid != 0 {
                        return Err("duplicate gadget ID".into());
                    }
                    source_gadgets[position] = value;
                }
                bin::instruction::Create::CheckModel(value) => {
                    let position = index(value.cid, count)?;
                    if value.cid != value.gid || source_checks[position].cid != 0 {
                        return Err("compiled-window requires exactly one check model with cid=gid".into());
                    }
                    if value.error_model_count.unwrap_or(1) != 1 {
                        return Err("compiled-window requires one error model per gadget".into());
                    }
                    source_checks[position] = value;
                }
                bin::instruction::Create::ErrorModel(value) => {
                    let position = index(value.eid, count)?;
                    if value.eid != value.cid || source_errors[position].eid != 0 {
                        return Err("compiled-window requires exactly one error model with eid=cid".into());
                    }
                    if value
                        .modifier
                        .as_ref()
                        .is_some_and(|modifier| modifier.probability_modifier.is_some())
                    {
                        return Err("compiled-window does not support probability modifiers".into());
                    }
                    source_errors[position] = value;
                }
            }
        }
        if source_gadgets.iter().any(|item| item.gid == 0)
            || source_checks.iter().any(|item| item.cid == 0)
            || source_errors.iter().any(|item| item.eid == 0)
        {
            return Err("compiled-window requires dense aligned gadget/check/error IDs".into());
        }
        let mut outputs = Vec::with_capacity(count);
        let mut offsets = Vec::with_capacity(count);
        let mut total = 0_usize;
        for gadget in &source_gadgets {
            let kind = gadget_types.get(&gadget.gtype).ok_or("unknown gadget type")?;
            outputs.push(vec![None; kind.outputs.len()]);
            offsets.push(total);
            total = total
                .checked_add(kind.measurements.len())
                .ok_or("measurement count overflow")?;
        }
        let mut frames = Vec::with_capacity(count);
        let mut frame_cache = HashMap::new();
        for (gid, gadget) in source_gadgets.iter().enumerate() {
            let kind = &gadget_types[&gadget.gtype];
            if gadget.connectors.len() != kind.inputs.len() {
                return Err("input port count mismatch".into());
            }
            let port_bits = |ptype| {
                port_types
                    .get(&ptype)
                    .map(|value| value.observables.len())
                    .ok_or("unknown port type")
            };
            let input_bits = kind
                .inputs
                .iter()
                .map(|port| port_bits(port.ptype))
                .collect::<Result<Vec<_>, _>>()?;
            let output_bits = kind
                .outputs
                .iter()
                .map(|port| port_bits(port.ptype))
                .collect::<Result<Vec<_>, _>>()?;
            let mut inputs = Vec::new();
            for (input, connector) in gadget.connectors.iter().enumerate() {
                let parent = index(connector.gid, count)?;
                if parent >= gid {
                    return Err("compiled frames require topologically ordered input IDs".into());
                }
                let port = usize::try_from(connector.port).map_err(|_| "port exceeds usize")?;
                let parent_kind = &gadget_types[&source_gadgets[parent].gtype];
                if parent_kind.outputs.get(port).map(|item| item.ptype) != Some(kind.inputs[input].ptype) {
                    return Err("connected port types do not match".into());
                }
                if outputs[parent][port].replace(gid).is_some() {
                    return Err("output port connected twice".into());
                }
                let offset: usize = parent_kind.outputs[..port]
                    .iter()
                    .map(|item| port_bits(item.ptype))
                    .collect::<Result<Vec<_>, _>>()?
                    .iter()
                    .sum();
                inputs.push(FrameInput {
                    gid: parent,
                    offset,
                    bits: input_bits[input],
                });
            }
            let mut normalized = gadget.modifier.clone().unwrap_or_default();
            let mut remote_readouts = Vec::new();
            if let Some(remote) = &mut normalized.remote_conditional_correction {
                for readout in &remote.remote_readouts {
                    let parent = index(readout.gid, count)?;
                    let bit = usize::try_from(readout.readout_index).map_err(|_| "readout index exceeds usize")?;
                    if parent >= gid || bit >= gadget_types[&source_gadgets[parent].gtype].readouts.len() {
                        return Err("remote readout must reference an earlier valid readout".into());
                    }
                    remote_readouts.push((parent, bit));
                }
                for readout in &mut remote.remote_readouts {
                    readout.gid = 0;
                    readout.readout_index = 0;
                }
            }
            let key = (gadget.gtype, normalized.encode_to_vec());
            let frame = if let Some(frame) = frame_cache.get(&key) {
                Arc::clone(frame)
            } else {
                let frame = Arc::new(FrameType::compile(
                    kind,
                    gadget.modifier.as_ref(),
                    input_bits.iter().sum(),
                    output_bits.iter().sum(),
                )?);
                frame_cache.insert(key, Arc::clone(&frame));
                frame
            };
            frames.push(GadgetPlan {
                frame,
                inputs,
                remote_readouts,
                measurement_start: offsets[gid],
                measurements: kind.measurements.len(),
            });
        }
        let mut compiled_checks = HashMap::new();
        for (&ctype, kind) in &check_types {
            let mut rows = Vec::new();
            let mut limits: HashMap<usize, usize> = HashMap::new();
            for check in &kind.checks {
                let terms = check
                    .measurements
                    .iter()
                    .map(|term| {
                        Ok((
                            term.remote_gadget.map_or(Ok(0), |remote| {
                                usize::try_from(remote)
                                    .map(|value| value + 1)
                                    .map_err(|_| "remote index overflow")
                            })?,
                            usize::try_from(term.measurement_index).map_err(|_| "measurement index overflow")?,
                        ))
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                for &(source, bit) in &terms {
                    limits
                        .entry(source)
                        .and_modify(|limit| *limit = (*limit).max(bit))
                        .or_insert(bit);
                }
                rows.push((check.naturally_flipped, terms));
            }
            compiled_checks.insert(
                ctype,
                Arc::new(CheckType {
                    id: ctype,
                    rows,
                    source_limits: limits.into_iter().collect(),
                }),
            );
        }
        let mut error_limits = HashMap::new();
        for (&etype, kind) in &error_types {
            let mut residual_bits = 0;
            let mut readout_bits = 0;
            let mut limits: HashMap<Option<usize>, usize> = HashMap::new();
            for error in &kind.errors {
                for &bit in &error.residual {
                    residual_bits = residual_bits.max(
                        usize::try_from(bit)
                            .ok()
                            .and_then(|bit| bit.checked_add(1))
                            .ok_or("residual index overflow")?,
                    );
                }
                for &bit in &error.readout_flips {
                    readout_bits = readout_bits.max(
                        usize::try_from(bit)
                            .ok()
                            .and_then(|bit| bit.checked_add(1))
                            .ok_or("readout index overflow")?,
                    );
                }
                for check in &error.checks {
                    let remote = check
                        .remote_check_model
                        .map(usize::try_from)
                        .transpose()
                        .map_err(|_| "remote check overflow")?;
                    let bit = usize::try_from(check.check_index).map_err(|_| "check index overflow")?;
                    limits
                        .entry(remote)
                        .and_modify(|limit| *limit = (*limit).max(bit))
                        .or_insert(bit);
                }
            }
            error_limits.insert(
                etype,
                ErrorLimits {
                    residual_bits,
                    readout_bits,
                    checks: limits.into_iter().collect(),
                },
            );
        }
        let mut checks = Vec::with_capacity(count);
        let mut errors = Vec::with_capacity(count);
        for gid in 0..count {
            let source = &source_checks[gid];
            let kind = check_types.get(&source.ctype).ok_or("unknown check type")?;
            if kind.gtype != 0 && kind.gtype != source_gadgets[gid].gtype {
                return Err("check/gadget type mismatch".into());
            }
            let remotes = apply_check_model_reroutes(&kind.remote_gadgets, source.modifier.as_ref())?;
            let mut resolved = Vec::new();
            let mut bases = vec![offsets[gid]];
            let mut lengths = vec![frames[gid].measurements];
            for (position, remote) in remotes.iter().enumerate() {
                let value = if let Some(remote) = remote {
                    let target = if let Some(absolute) = remote.absolute_gid {
                        Some(index(absolute, count)?)
                    } else {
                        let previous = if let Some(previous) = remote.previous_remote_gadget {
                            let previous = usize::try_from(previous).map_err(|_| "remote predecessor overflow")?;
                            if previous >= position {
                                return Err("remote paths must refer to earlier path entries".into());
                            }
                            resolved[previous]
                        } else {
                            Some(gid)
                        };
                        let port = remote.port.map(|value| match value {
                            bin::check_model_type::remote_gadget::Port::Input(port) => (false, port),
                            bin::check_model_type::remote_gadget::Port::Output(port) => (true, port),
                        });
                        hop(previous, port, &source_gadgets, &outputs)?
                    };
                    if let Some(target) = target {
                        let bias = usize::try_from(remote.measurement_bias).map_err(|_| "measurement bias overflow")?;
                        if bias > frames[target].measurements {
                            return Err("measurement bias out of range".into());
                        }
                        bases.push(offsets[target] + bias);
                        lengths.push(frames[target].measurements - bias);
                    } else {
                        bases.push(0);
                        lengths.push(0);
                    }
                    target
                } else {
                    bases.push(0);
                    lengths.push(0);
                    None
                };
                resolved.push(value);
            }
            let compiled = Arc::clone(&compiled_checks[&source.ctype]);
            let mut last_input = gid;
            for &(source_index, measurement) in &compiled.source_limits {
                if measurement >= *lengths.get(source_index).ok_or("check remote index out of range")? {
                    return Err("check references an unavailable measurement".into());
                }
                if source_index != 0 {
                    last_input = last_input.max(resolved[source_index - 1].ok_or("unresolved check input")?);
                }
            }
            checks.push(CheckPlan {
                kind: compiled,
                bases,
                last_input,
            });
            let source = &source_errors[gid];
            let kind = error_types.get(&source.etype).ok_or("unknown error type")?;
            if kind.ctype != 0 && kind.ctype != source_checks[gid].ctype {
                return Err("error/check type mismatch".into());
            }
            let modified = apply_error_model_reroutes(&kind.remote_check_models, source.modifier.as_ref())?;
            let mut remotes: Vec<Option<(usize, usize)>> = Vec::new();
            for (position, remote) in modified.iter().enumerate() {
                let target = if let Some(remote) = remote {
                    let target = if let Some(absolute) = remote.absolute_cid {
                        // The full model may explicitly name an external boundary.
                        if absolute == u64::MAX - 2 {
                            None
                        } else {
                            Some(index(absolute, count)?)
                        }
                    } else {
                        let previous = if let Some(previous) = remote.previous_remote_check_model {
                            let previous = usize::try_from(previous).map_err(|_| "remote predecessor overflow")?;
                            if previous >= position {
                                return Err("remote check paths must refer to earlier path entries".into());
                            }
                            remotes[previous].map(|value| value.0)
                        } else {
                            Some(gid)
                        };
                        let port = remote.port.map(|value| match value {
                            bin::error_model_type::remote_check_model::Port::Input(port) => (false, port),
                            bin::error_model_type::remote_check_model::Port::Output(port) => (true, port),
                        });
                        hop(previous, port, &source_gadgets, &outputs)?
                    };
                    target.map(|target| (target, usize::try_from(remote.check_bias).unwrap()))
                } else {
                    None
                };
                remotes.push(target);
            }
            let mut support = vec![gid];
            let limits = &error_limits[&kind.etype];
            if limits.residual_bits > frames[gid].frame.output_bits || limits.readout_bits > frames[gid].frame.readouts.len()
            {
                return Err("error frame correction is out of range".into());
            }
            for &(remote, bit) in &limits.checks {
                let target = if let Some(remote) = remote {
                    *remotes.get(remote).ok_or("remote error check is out of range")?
                } else {
                    Some((gid, 0))
                };
                if let Some((target, bias)) = target {
                    let target_type = &check_types[&source_checks[target].ctype];
                    if bias.checked_add(bit).is_none_or(|bit| bit >= target_type.checks.len()) {
                        return Err("error references a check outside its type".into());
                    }
                    if !support.contains(&target) {
                        support.push(target);
                    }
                }
            }
            support.sort_unstable();
            errors.push(ErrorPlan {
                kind: Arc::clone(kind),
                remotes,
                support,
            });
        }
        drop(source_gadgets);
        drop(source_checks);
        drop(source_errors);
        drop(outputs);
        let mut incident_edges = Vec::new();
        for (gid, error) in errors.iter().enumerate() {
            if !error.kind.errors.is_empty() {
                incident_edges.extend(error.support.iter().map(|&cid| (cid, gid)));
            }
        }
        let incident = Adjacency::build(count, &incident_edges);
        drop(incident_edges);
        let mut templates = HashMap::<WindowKey, Arc<Template>>::new();
        let mut windows = Vec::with_capacity(count);
        for gid in 0..count {
            if errors[gid].kind.errors.is_empty() {
                windows.push(None);
                continue;
            }
            let read_checks = errors[gid].support.clone();
            let mut owners: Vec<_> = read_checks
                .iter()
                .flat_map(|&cid| incident.get(cid).iter().copied())
                .collect();
            owners.sort_unstable();
            owners.dedup();
            let predecessors: Vec<_> = owners.iter().copied().filter(|&owner| owner < gid).collect();
            let patterns = owners
                .iter()
                .copied()
                .filter(|&owner| owner >= gid)
                .map(|owner| ErrorPattern {
                    etype: errors[owner].kind.etype,
                    local: read_checks.binary_search(&owner).ok(),
                    remotes: errors[owner]
                        .remotes
                        .iter()
                        .map(|remote| {
                            remote.and_then(|(cid, bias)| read_checks.binary_search(&cid).ok().map(|index| (index, bias)))
                        })
                        .collect(),
                    owned: owner == gid,
                })
                .collect();
            let key = WindowKey {
                checks: read_checks.iter().map(|&cid| checks[cid].kind.id).collect(),
                errors: patterns,
            };
            let template = if let Some(template) = templates.get(&key) {
                Arc::clone(template)
            } else {
                if templates.len() == max_templates {
                    return Err("compiled decoder template limit exceeded".into());
                }
                let template = Arc::new(Self::template(&key, &check_types, &error_types));
                templates.insert(key, Arc::clone(&template));
                template
            };
            let last_input = read_checks.iter().map(|&cid| checks[cid].last_input).max().unwrap();
            windows.push(Some(WindowPlan {
                template,
                checks: read_checks,
                predecessors,
                last_input,
            }));
        }
        let mut edges = Vec::new();
        let mut raw_edges = Vec::new();
        let mut remaining = vec![0_u32; count * 3];
        for gid in 0..count {
            if !checks[gid].kind.rows.is_empty() {
                raw_edges.push((checks[gid].last_input, gid * 3 + SYNDROME));
                remaining[gid * 3 + SYNDROME] += 1;
            }
            if let Some(window) = &windows[gid] {
                let mut prerequisites: Vec<_> = window
                    .checks
                    .iter()
                    .filter(|&&cid| !checks[cid].kind.rows.is_empty())
                    .map(|&cid| cid * 3 + SYNDROME)
                    .collect();
                prerequisites.extend(
                    window
                        .predecessors
                        .iter()
                        .filter(|&&owner| windows[owner].is_some())
                        .map(|&owner| owner * 3 + WINDOW),
                );
                // Even an empty check window must not commit before its own input arrives.
                raw_edges.push((window.last_input, gid * 3 + WINDOW));
                remaining[gid * 3 + WINDOW] += 1;
                prerequisites.sort_unstable();
                prerequisites.dedup();
                for prerequisite in prerequisites {
                    edges.push((prerequisite, gid * 3 + WINDOW));
                    remaining[gid * 3 + WINDOW] += 1;
                }
            }
            let mut prerequisites: Vec<_> = frames[gid].inputs.iter().map(|input| input.gid * 3 + FRAME).collect();
            prerequisites.extend(frames[gid].remote_readouts.iter().map(|&(parent, _)| parent * 3 + FRAME));
            if windows[gid].is_some() {
                prerequisites.push(gid * 3 + WINDOW);
            }
            prerequisites.sort_unstable();
            prerequisites.dedup();
            for prerequisite in prerequisites {
                edges.push((prerequisite, gid * 3 + FRAME));
                remaining[gid * 3 + FRAME] += 1;
            }
            raw_edges.push((gid, gid * 3 + FRAME));
            remaining[gid * 3 + FRAME] += 1;
        }
        Ok(Self {
            gadgets: frames,
            checks,
            errors,
            windows,
            templates: templates.into_values().collect(),
            dependents: Adjacency::build(count * 3, &edges),
            input_dependents: Adjacency::build(count, &raw_edges),
            remaining,
            total_measurements: total,
        })
    }

    fn template(
        key: &WindowKey,
        checks: &HashMap<u64, bin::CheckModelType>,
        errors: &HashMap<u64, Arc<bin::ErrorModelType>>,
    ) -> Template {
        let mut starts = Vec::new();
        let mut vertices = 0;
        for ctype in &key.checks {
            starts.push(vertices);
            vertices += checks[ctype].checks.len();
        }
        let mut hyperedges = Vec::new();
        let mut references = Vec::new();
        for (owner, pattern) in key.errors.iter().enumerate() {
            for (error_index, error) in errors[&pattern.etype].errors.iter().enumerate() {
                let mut incidence = Vec::new();
                let mut outside = false;
                for check in &error.checks {
                    let target = if let Some(remote) = check.remote_check_model {
                        pattern.remotes[usize::try_from(remote).unwrap()]
                    } else {
                        pattern.local.map(|local| (local, 0))
                    };
                    if let Some((local, bias)) = target {
                        incidence.push(u64::try_from(starts[local] + bias).unwrap() + check.check_index);
                    } else {
                        outside = true;
                    }
                }
                if pattern.owned && outside {
                    continue;
                }
                if incidence.is_empty() {
                    continue;
                }
                hyperedges.push(Hyperedge {
                    vertices: incidence,
                    probability: error.probability,
                    observable_flips: None,
                });
                references.push(ErrorIndex { eid: owner, error_index });
            }
        }
        let graph = DecodingHypergraph {
            vertex_num: u64::try_from(vertices).unwrap(),
            hyperedges,
        };
        let flips = vec![Vec::new(); references.len()];
        let prepared =
            deduplicate_decoder_input(&graph, &references, &flips, |error| usize::from(key.errors[error.eid].owned));
        let edge_errors = prepared
            .representatives
            .iter()
            .map(|error| key.errors[error.eid].owned.then_some(error.error_index))
            .collect();
        let isolated = edge_isolated_vertices(&prepared.hypergraph)
            .into_iter()
            .map(|value| usize::try_from(value).unwrap())
            .collect();
        Template {
            graph: Arc::new(SharedDecodingHypergraph::new(prepared.hypergraph)),
            edge_errors,
            isolated,
            starts,
            hid: OnceLock::new(),
        }
    }
}
