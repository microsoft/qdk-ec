use super::CompiledWindowConfig;
use super::frame::bit_mask;
use super::plan::{FRAME, Plan, SYNDROME, WINDOW};
use crate::coordinator::{self, Readouts};
use crate::decoder::{DynDecoder, blackbox_decoder};
use crate::misc::bit_vector;
use crate::misc::sync::TaskCounter;
use crate::util::BitVector;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::{Arc, OnceLock};
use tokio::sync::{mpsc, oneshot};
use tonic::Status;

pub(super) struct Syndrome {
    bits: Vec<u8>,
    count: u64,
}

#[derive(Default)]
pub(super) struct Correction {
    residual: u128,
    readouts: u128,
    flips: Vec<(usize, usize)>,
    count: u64,
    weight: f64,
    required_input: usize,
}

pub(super) struct Frame {
    residual: u128,
    readouts: u128,
    required_input: usize,
}

pub(super) struct Epoch {
    pub cancelled: AtomicBool,
    pub plan: Arc<Plan>,
    raw: Vec<AtomicU8>,
    syndromes: Vec<OnceLock<Syndrome>>,
    corrections: Vec<OnceLock<Correction>>,
    frames: Vec<OnceLock<Frame>>,
    jobs: Arc<TaskCounter>,
}

impl Epoch {
    fn new(plan: Arc<Plan>) -> Self {
        let count = plan.gadgets.len();
        Self {
            raw: (0..plan.total_measurements.div_ceil(8)).map(|_| AtomicU8::new(0)).collect(),
            syndromes: (0..count).map(|_| OnceLock::new()).collect(),
            corrections: (0..count).map(|_| OnceLock::new()).collect(),
            frames: (0..count).map(|_| OnceLock::new()).collect(),
            jobs: TaskCounter::new(),
            cancelled: AtomicBool::new(false),
            plan,
        }
    }

    fn raw_bit(&self, index: usize) -> bool {
        self.raw[index / 8].load(Ordering::Relaxed) & (0x80 >> (index % 8)) != 0
    }

    #[allow(
        clippy::too_many_lines,
        reason = "The three bounded dataflow task variants share only immutable epoch state."
    )]
    fn run(
        &self,
        task: usize,
        decoder: &DynDecoder,
        runtime: &tokio::runtime::Handle,
        seed: Option<u64>,
    ) -> Result<(), Status> {
        if self.cancelled.load(Ordering::Acquire) {
            return Err(Status::cancelled("compiled decoding cancelled"));
        }
        let gid = task / 3;
        match task % 3 {
            SYNDROME => {
                let plan = &self.plan.checks[gid];
                let mut bits = vec![0; plan.kind.rows.len().div_ceil(8)];
                let mut count = 0;
                for (index, (bias, terms)) in plan.kind.rows.iter().enumerate() {
                    let value = terms
                        .iter()
                        .fold(*bias, |value, &(source, bit)| value ^ self.raw_bit(plan.bases[source] + bit));
                    if value {
                        bits[index / 8] |= 0x80 >> (index % 8);
                        count += 1;
                    }
                }
                self.syndromes[gid]
                    .set(Syndrome { bits, count })
                    .map_err(|_| Status::internal("syndrome task ran twice"))?;
            }
            WINDOW => {
                let plan = self.plan.windows[gid].as_ref().unwrap();
                let template = &plan.template;
                let mut syndrome = BitVector {
                    size: template.graph.vertex_num,
                    data: vec![0; usize::try_from(template.graph.vertex_num).unwrap().div_ceil(8)],
                };
                for (local, &cid) in plan.checks.iter().enumerate() {
                    if let Some(source) = self.syndromes[cid].get() {
                        for bit in 0..self.plan.checks[cid].kind.rows.len() {
                            if source.bits[bit / 8] & (0x80 >> (bit % 8)) != 0 {
                                bit_vector::flip_bit(&mut syndrome, u64::try_from(template.starts[local] + bit).unwrap());
                            }
                        }
                    } else if !self.plan.checks[cid].kind.rows.is_empty() {
                        return Err(Status::internal("window ran before syndrome readiness"));
                    }
                }
                let mut correction = Correction {
                    required_input: plan.last_input,
                    ..Default::default()
                };
                for &predecessor in &plan.predecessors {
                    if let Some(previous) = self.corrections[predecessor].get() {
                        correction.required_input = correction.required_input.max(previous.required_input);
                        for &(cid, bit) in &previous.flips {
                            if let Ok(local) = plan.checks.binary_search(&cid) {
                                bit_vector::flip_bit(&mut syndrome, u64::try_from(template.starts[local] + bit).unwrap());
                            }
                        }
                    } else if self.plan.windows[predecessor].is_some() {
                        return Err(Status::internal("window ran before correction readiness"));
                    }
                }
                for &vertex in &template.isolated {
                    let owner = template.starts.partition_point(|&start| start <= vertex).saturating_sub(1);
                    if plan.checks[owner] >= gid && bit_vector::get_bit(&syndrome, u64::try_from(vertex).unwrap()) {
                        return Err(Status::failed_precondition(
                            "active syndrome has no modelled error in its compiled window",
                        ));
                    }
                    bit_vector::set_bit(&mut syndrome, u64::try_from(vertex).unwrap(), false);
                }
                if !bit_vector::is_zero(&syndrome) {
                    let result = runtime.block_on(
                        decoder.decode_loaded(blackbox_decoder::LoadedDecodingProblem {
                            hid: *template
                                .hid
                                .get()
                                .ok_or_else(|| Status::internal("decoder template not loaded"))?,
                            syndrome: Some(syndrome.clone()),
                            decoder_seed: seed,
                            ..Default::default()
                        }),
                    )?;
                    let mut verified = BitVector {
                        size: syndrome.size,
                        data: vec![0; syndrome.data.len()],
                    };
                    for &edge in &result.subgraph {
                        let edge = usize::try_from(edge).map_err(|_| Status::internal("decoder edge overflow"))?;
                        let hyperedge = template
                            .graph
                            .hyperedges
                            .get(edge)
                            .ok_or_else(|| Status::internal("decoder returned an invalid edge"))?;
                        for &vertex in &hyperedge.vertices {
                            bit_vector::flip_bit(&mut verified, vertex);
                        }
                        if let Some(error) = template.edge_errors[edge] {
                            let source = &self.plan.errors[gid];
                            let error = &source.kind.errors[error];
                            correction.count += 1;
                            correction.weight += crate::misc::util::weight_of(hyperedge.probability);
                            for &bit in &error.residual {
                                correction.residual ^= 1_u128 << bit;
                            }
                            for &bit in &error.readout_flips {
                                correction.readouts ^= 1_u128 << bit;
                            }
                            for check in &error.checks {
                                let (cid, bias) = check
                                    .remote_check_model
                                    .map_or(Some((gid, 0)), |remote| source.remotes[usize::try_from(remote).unwrap()])
                                    .ok_or_else(|| Status::internal("committed correction escaped its compiled window"))?;
                                correction
                                    .flips
                                    .push((cid, bias + usize::try_from(check.check_index).unwrap()));
                            }
                        }
                    }
                    if verified != syndrome {
                        return Err(Status::internal("decoder correction does not match the compiled syndrome"));
                    }
                }
                self.corrections[gid]
                    .set(correction)
                    .map_err(|_| Status::internal("window task ran twice"))?;
            }
            FRAME => self.compute_frame(gid)?,
            _ => unreachable!(),
        }
        Ok(())
    }

    fn compute_frame(&self, gid: usize) -> Result<(), Status> {
        let plan = &self.plan.gadgets[gid];
        let kind = &plan.frame;
        let mut required_input = gid;
        let mut input = 0_u128;
        let mut offset = 0;
        for connection in &plan.inputs {
            let frame = self.frames[connection.gid]
                .get()
                .ok_or_else(|| Status::internal("frame input not ready"))?;
            required_input = required_input.max(frame.required_input);
            input |= ((frame.residual >> connection.offset) & bit_mask(connection.bits)) << offset;
            offset += connection.bits;
        }
        input |= 1_u128 << kind.input_bits;
        let mut remote = 0_u128;
        for (index, &(parent, bit)) in plan.remote_readouts.iter().enumerate() {
            let frame = self.frames[parent]
                .get()
                .ok_or_else(|| Status::internal("remote frame input not ready"))?;
            required_input = required_input.max(frame.required_input);
            remote |= ((frame.readouts >> bit) & 1) << index;
        }
        let mut readouts = 0_u128;
        for (index, measurements) in kind.readouts.iter().enumerate() {
            if measurements
                .iter()
                .fold(false, |value, &bit| value ^ self.raw_bit(plan.measurement_start + bit))
            {
                readouts |= 1_u128 << index;
            }
        }
        let mut residual = kind.input_to_output.apply(input)
            ^ kind
                .physical_to_output
                .apply_raw(|bit| self.raw_bit(plan.measurement_start + bit));
        if let Some(correction) = self.corrections[gid].get() {
            required_input = required_input.max(correction.required_input);
            residual ^= correction.residual;
            readouts ^= correction.readouts;
        } else if self.plan.windows[gid].is_some() {
            return Err(Status::internal("frame correction not ready"));
        }
        readouts ^= kind.input_to_readout.apply(input);
        residual ^= kind.readout_to_output.apply(readouts) ^ kind.remote_to_output.apply(remote);
        self.frames[gid]
            .set(Frame {
                residual,
                readouts,
                required_input,
            })
            .map_err(|_| Status::internal("frame task ran twice"))?;
        #[cfg(feature = "cli")]
        if !kind.readouts.is_empty()
            && let Some(mut trace) = crate::misc::fastrace::LatencySpan::new("compiled_readout")
        {
            let ready = std::time::Instant::now();
            trace.property("gid", (gid + 1).to_string());
            trace.property("input_gid", (required_input + 1).to_string());
            trace.property("readout_count", kind.readouts.len().to_string());
            trace.event_at("ready", ready);
        }
        Ok(())
    }

    pub fn readout_bits(&self, gid: u64) -> Result<Option<BitVector>, Status> {
        if self.cancelled.load(Ordering::Acquire) {
            return Err(Status::cancelled("compiled epoch is cancelled; reset required"));
        }
        let index = gid
            .checked_sub(1)
            .and_then(|index| usize::try_from(index).ok())
            .filter(|&index| index < self.frames.len())
            .ok_or_else(|| Status::invalid_argument("gadget ID outside the compiled program"))?;
        Ok(self.frames[index].get().map(|frame| self.frame_bits(index, frame)))
    }

    fn frame_bits(&self, gid: usize, frame: &Frame) -> BitVector {
        let count = self.plan.gadgets[gid].frame.readouts.len();
        let mut bits = BitVector {
            size: u64::try_from(count).unwrap(),
            data: vec![0; count.div_ceil(8)],
        };
        for bit in 0..count {
            bit_vector::set_bit(&mut bits, u64::try_from(bit).unwrap(), frame.readouts & (1_u128 << bit) != 0);
        }
        bits
    }

    fn readouts(&self) -> Result<Vec<Readouts>, Status> {
        self.plan
            .gadgets
            .iter()
            .enumerate()
            .map(|(gid, _)| {
                let frame = self.frames[gid]
                    .get()
                    .ok_or_else(|| Status::internal("compiled frame missing"))?;
                let correction = self.corrections[gid].get();
                Ok(Readouts {
                    gid: u64::try_from(gid + 1).unwrap(),
                    readouts: Some(self.frame_bits(gid, frame)),
                    syndrome_count: self.syndromes[gid].get().map_or(0, |value| value.count),
                    correction_count: correction.map_or(0, |value| value.count),
                    correction_weight: correction.map_or(0.0, |value| value.weight),
                    ..Default::default()
                })
            })
            .collect()
    }
}

enum Command {
    Input {
        outcomes: BitVector,
        through: Option<usize>,
        seed: Option<u64>,
        reply: oneshot::Sender<Result<Vec<Readouts>, Status>>,
    },
    Completed(Vec<(usize, Result<(), Status>)>),
    Cancel,
    Stop(oneshot::Sender<()>),
}

struct InputGuard<'a> {
    session: &'a Session,
    armed: bool,
}

impl Drop for InputGuard<'_> {
    fn drop(&mut self) {
        if self.armed {
            self.session.epoch.cancelled.store(true, Ordering::Release);
            let _ = self.session.sender.try_send(Command::Cancel);
        }
    }
}

pub(super) struct Session {
    pub epoch: Arc<Epoch>,
    sender: mpsc::Sender<Command>,
}

impl Drop for Session {
    fn drop(&mut self) {
        self.epoch.cancelled.store(true, Ordering::Release);
    }
}

impl Session {
    #[allow(
        clippy::too_many_lines,
        reason = "A single owner handles admission, dependency release, and draining without cross-lock state."
    )]
    pub fn new(
        plan: Arc<Plan>,
        decoder: DynDecoder,
        pool: Arc<rayon::ThreadPool>,
        config: &CompiledWindowConfig,
    ) -> Result<Arc<Self>, Status> {
        let epoch = Arc::new(Epoch::new(plan));
        let (sender, mut receiver) = mpsc::channel(128);
        let completion = sender.downgrade();
        let runtime = tokio::runtime::Handle::current();
        let worker_threads = config.worker_threads;
        let batch_size = config.batch_size;
        let session = Arc::new(Self {
            epoch: Arc::clone(&epoch),
            sender,
        });
        std::thread::Builder::new()
            .name("deq-compiled".into())
            .spawn(move || {
                let mut remaining = epoch.plan.remaining.clone();
                let mut ready: [VecDeque<usize>; 3] = std::array::from_fn(|_| VecDeque::new());
                let mut input = BitVector {
                    size: 0,
                    data: Vec::with_capacity(epoch.raw.len()),
                };
                let mut published = 0;
                let mut seed = None;
                let mut running = 0;
                let mut finished = 0;
                let total = epoch.plan.gadgets.len()
                    + epoch.plan.checks.iter().filter(|check| !check.kind.rows.is_empty()).count()
                    + epoch.plan.windows.iter().flatten().count();
                let mut pending_reply = None;
                let mut stopping = None;
                let mut failure: Option<Status> = None;
                while let Some(command) = receiver.blocking_recv() {
                    match command {
                        Command::Input {
                            outcomes,
                            through,
                            seed: requested_seed,
                            reply,
                        } => {
                            let result = (|| {
                                if epoch.cancelled.load(Ordering::Acquire) || failure.is_some() {
                                    return Err(Status::cancelled("compiled epoch is cancelled; reset required"));
                                }
                                if published == epoch.plan.gadgets.len() {
                                    return Err(Status::already_exists("compiled shot already submitted; reset required"));
                                }
                                let new_size = input
                                    .size
                                    .checked_add(outcomes.size)
                                    .ok_or_else(|| Status::invalid_argument("measurement count overflow"))?;
                                if new_size > u64::try_from(epoch.plan.total_measurements).unwrap() {
                                    return Err(Status::invalid_argument("too many measurements"));
                                }
                                let through = through.unwrap_or_else(|| {
                                    epoch.plan.gadgets.partition_point(|gadget| {
                                        u64::try_from(gadget.measurement_start + gadget.measurements).unwrap() <= new_size
                                    })
                                });
                                if through < published || through > epoch.plan.gadgets.len() {
                                    return Err(Status::invalid_argument("invalid gadget completion boundary"));
                                }
                                let expected = through.checked_sub(1).map_or(0, |gid| {
                                    epoch.plan.gadgets[gid].measurement_start + epoch.plan.gadgets[gid].measurements
                                });
                                if new_size != u64::try_from(expected).unwrap() {
                                    return Err(Status::invalid_argument(
                                        "outcomes do not match the compiled completion boundary",
                                    ));
                                }
                                coordinator::accept_decoder_seed(&mut seed, requested_seed)?;
                                let start = usize::try_from(input.size / 8).unwrap();
                                bit_vector::append(&mut input, &outcomes);
                                for (target, &byte) in epoch.raw[start..input.data.len()].iter().zip(&input.data[start..]) {
                                    target.store(byte, Ordering::Relaxed);
                                }
                                for gid in published..through {
                                    for &task in epoch.plan.input_dependents.get(gid) {
                                        remaining[task] -= 1;
                                        if remaining[task] == 0 {
                                            ready[task % 3].push_back(task);
                                        }
                                    }
                                }
                                published = through;
                                Ok(through == epoch.plan.gadgets.len())
                            })();
                            match result {
                                Ok(true) if pending_reply.is_none() => pending_reply = Some(reply),
                                Ok(false) => {
                                    let _ = reply.send(Ok(Vec::new()));
                                }
                                Ok(true) => {
                                    let _ = reply.send(Err(Status::already_exists("compiled shot was already submitted")));
                                }
                                Err(error) => {
                                    let _ = reply.send(Err(error));
                                }
                            }
                        }
                        Command::Completed(completed) => {
                            running -= 1;
                            for (task, result) in completed {
                                match result {
                                    Ok(()) => {
                                        finished += 1;
                                        for &dependent in epoch.plan.dependents.get(task) {
                                            remaining[dependent] -= 1;
                                            if remaining[dependent] == 0 {
                                                ready[dependent % 3].push_back(dependent);
                                            }
                                        }
                                    }
                                    Err(error) => {
                                        failure.get_or_insert(error);
                                        epoch.cancelled.store(true, Ordering::Release);
                                    }
                                }
                            }
                        }
                        Command::Stop(stopped) => {
                            epoch.cancelled.store(true, Ordering::Release);
                            stopping = Some(stopped);
                        }
                        Command::Cancel => epoch.cancelled.store(true, Ordering::Release),
                    }
                    if epoch.cancelled.load(Ordering::Acquire) {
                        if let Some(reply) = pending_reply.take() {
                            let _ = reply.send(Err(failure
                                .clone()
                                .unwrap_or_else(|| Status::cancelled("compiled epoch cancelled"))));
                        }
                        if running == 0 {
                            if let Some(stopped) = stopping.take() {
                                let _ = stopped.send(());
                            }
                            break;
                        }
                        continue;
                    }
                    if finished == total
                        && let Some(reply) = pending_reply.take()
                    {
                        let _ = reply.send(epoch.readouts());
                    }
                    while running < worker_threads * 2 && !epoch.cancelled.load(Ordering::Acquire) {
                        let mut tasks = Vec::with_capacity(batch_size);
                        for kind in [FRAME, WINDOW, SYNDROME] {
                            while tasks.len() < batch_size {
                                let Some(task) = ready[kind].pop_front() else { break };
                                tasks.push(task);
                            }
                        }
                        if tasks.is_empty() {
                            break;
                        }
                        let Some(sender) = completion.upgrade() else { break };
                        let epoch = Arc::clone(&epoch);
                        let decoder = decoder.clone();
                        let runtime = runtime.clone();
                        let seed = seed.flatten();
                        let tracked = epoch.jobs.guard();
                        running += 1;
                        pool.spawn(move || {
                            let _tracked = tracked;
                            let results = tasks
                                .into_iter()
                                .map(|task| {
                                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                        epoch.run(task, &decoder, &runtime, seed)
                                    }))
                                    .unwrap_or_else(|_| Err(Status::internal("compiled worker panicked")));
                                    (task, result)
                                })
                                .collect();
                            let _ = sender.blocking_send(Command::Completed(results));
                        });
                    }
                }
            })
            .map_err(|error| Status::internal(format!("cannot start compiled coordinator: {error}")))?;
        Ok(session)
    }

    pub async fn ingest(
        &self,
        outcomes: BitVector,
        through: Option<usize>,
        seed: Option<u64>,
    ) -> Result<Vec<Readouts>, Status> {
        let mut guard = InputGuard {
            session: self,
            armed: true,
        };
        let (reply, received) = oneshot::channel();
        self.sender
            .send(Command::Input {
                outcomes,
                through,
                seed,
                reply,
            })
            .await
            .map_err(|_| Status::unavailable("compiled coordinator stopped"))?;
        let result = received
            .await
            .map_err(|_| Status::internal("compiled input request abandoned"))?;
        guard.armed = false;
        result
    }

    pub async fn stop(&self) {
        self.epoch.cancelled.store(true, Ordering::Release);
        let (stopped, received) = oneshot::channel();
        if self.sender.send(Command::Stop(stopped)).await.is_ok() {
            let _ = received.await;
        }
        self.epoch.jobs.wait_for_zero().await;
    }
}
