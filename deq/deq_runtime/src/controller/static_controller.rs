use crate::bin;
use crate::coordinator;
use crate::coordinator::{CoordinatorClient, ResetRequest};
use crate::misc::bit_vector;
use crate::misc::fastrace::{Span, SpanContext};
use crate::util::BitVector;
use serde::{Deserialize, Serialize};
use std::num::NonZeroUsize;
use std::sync::Arc;
#[cfg(feature = "cli")]
use structdoc::StructDoc;
use tokio::sync::{Mutex, RwLock};
use tokio::task::JoinSet;
use tonic::Status;
#[cfg(feature = "cli")]
use tonic::transport::server::Router;
#[cfg(feature = "cli")]
use tonic::{Request, Response};

include!("../proto/deq.controller.static_controller.rs");

const DECODE_TASK_BATCH_SIZE: usize = 16;

fn default_decode_task_batch_size() -> usize {
    DECODE_TASK_BATCH_SIZE
}

fn deserialize_task_batch_size<'de, D: serde::Deserializer<'de>>(deserializer: D) -> Result<usize, D::Error> {
    NonZeroUsize::deserialize(deserializer).map(NonZeroUsize::get)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "cli", derive(StructDoc))]
#[serde(deny_unknown_fields)]
pub struct StaticControllerConfig {
    /// the filepath to the binary protobuf file, see `proto/coordinator.proto`
    pub filepath: String,
    #[serde(default)]
    pub reset_library: bool,
    #[serde(default)]
    pub reset_decoder_service: bool,
    /// Maximum pending gadget futures per Tokio task (default 16); must be positive.
    /// One gives each gadget its own task.
    #[serde(
        default = "default_decode_task_batch_size",
        deserialize_with = "deserialize_task_batch_size"
    )]
    pub decode_task_batch_size: usize,
}

pub struct StaticController {
    pub config: StaticControllerConfig,
    pub library: crate::bin::Library,
    coordinator: RwLock<Option<CoordinatorClient>>,
    /// immutable information about the library
    #[cfg_attr(not(feature = "cli"), allow(dead_code))]
    info: Arc<StaticControllerInfo>,
    /// mutable state during runtime
    #[cfg_attr(not(feature = "cli"), allow(dead_code))]
    state: Arc<Mutex<StaticControllerState>>,
}

struct StaticControllerState {
    /// the next gid to be instantiated
    next_index: usize,
    /// the list of gids instantiated
    gid_vec: Vec<u64>,
    /// accumulated measurement outcomes received
    outcomes: BitVector,
    /// Accumulated per-measurement loss flags, kept in lockstep with `outcomes`.
    loss_mask: Option<BitVector>,
    /// Seed selected by the first request in a shot. The outer option records whether
    /// that request arrived; the inner option preserves an absent seed.
    decoder_seed: Option<Option<u64>>,
    /// background decode tasks for streaming mode (dispatched but not yet awaited)
    pending_decodes: JoinSet<Result<Vec<(usize, coordinator::Readouts)>, Status>>,
    /// readouts collected from completed background decode tasks
    pending_readouts: Vec<Option<coordinator::Readouts>>,
}

struct StaticControllerInfo {
    /// the accumulated number of measurements, helpful to determine if sufficient
    /// number of measurements outcome bits have been received
    accumulated_measurements: Vec<usize>,
    /// the number of measurements in total
    total_measurements: usize,
}

impl StaticControllerInfo {
    pub fn new(library: &bin::Library) -> Self {
        let mut accumulated_measurements: Vec<usize> = vec![];
        let mut total_measurements: usize = 0;
        for instruction in library.program.iter() {
            if let bin::instruction::Create::Gadget(gadget) = instruction.create.as_ref().unwrap() {
                let gadget_type = library.gadget_types.iter().find(|gt| gt.gtype == gadget.gtype).unwrap();
                total_measurements += gadget_type.measurements.len();
                accumulated_measurements.push(total_measurements);
            }
        }
        Self {
            accumulated_measurements,
            total_measurements,
        }
    }
}

impl StaticController {
    pub fn new(config: serde_json::Value) -> Self {
        let _span = Span::root("static_library_load", SpanContext::random());
        let config: StaticControllerConfig = serde_json::from_value(config).unwrap();
        // read from file
        let data = std::fs::read(config.filepath.clone()).unwrap();
        let library: bin::Library = prost::Message::decode(&mut data.as_slice()).unwrap();
        let info = Arc::new(StaticControllerInfo::new(&library));
        let state = StaticControllerState {
            next_index: 0,
            gid_vec: Vec::with_capacity(info.accumulated_measurements.len()),
            outcomes: BitVector {
                size: 0,
                data: Vec::with_capacity(bit_vector::bit_vector_len(u64::try_from(info.total_measurements).unwrap())),
            },
            loss_mask: None,
            decoder_seed: None,
            pending_decodes: JoinSet::new(),
            pending_readouts: Vec::new(),
        };
        Self {
            config,
            library,
            coordinator: RwLock::new(None),
            info,
            state: Arc::new(Mutex::new(state)),
        }
    }

    #[cfg(feature = "cli")]
    pub fn add_service(self: &Arc<Self>, router: Router) -> Router {
        let service =
            static_controller_server::StaticControllerServer::from_arc(self.clone()).max_decoding_message_size(usize::MAX);
        router.add_service(service)
    }

    pub async fn start(self: &Arc<Self>, mut client: CoordinatorClient) {
        let mut coordinator = self.coordinator.write().await;
        client.load_library(self.library.clone()).await.unwrap();
        self.reset_and_instantiate_all(&mut client).await;
        coordinator.replace(client);
    }

    #[cfg(feature = "cli")]
    async fn wait_until_library_loaded(&self) -> CoordinatorClient {
        // tokio::sync::RwLock is a fair lock, so looping is fine
        while self.coordinator.read().await.is_none() {
            tokio::task::yield_now().await;
        }
        self.coordinator.read().await.clone().unwrap()
    }

    async fn reset_and_instantiate_all(&self, coordinator: &mut CoordinatorClient) {
        let _span = Span::root("static_program_load", SpanContext::random());
        // since a static program should never change, we reset the coordinator to a minimum extent
        coordinator
            .reset(ResetRequest {
                reset_library: self.config.reset_library,
                reset_decoder_service: self.config.reset_decoder_service,
                ..Default::default()
            })
            .await
            .unwrap();
        self.state.lock().await.pending_decodes.shutdown().await;
        let mut gid_vec: Vec<u64> = vec![];
        let mut cid_vec: Vec<u64> = vec![];
        for instruction in self.library.program.iter() {
            if matches!(
                coordinator,
                CoordinatorClient::Local(crate::coordinator::DynCoordinator::CompiledWindow(_))
            ) {
                if let Some(bin::instruction::Create::Gadget(gadget)) = &instruction.create {
                    gid_vec.push(gadget.gid);
                }
                continue;
            }
            match instruction.clone().create.unwrap() {
                bin::instruction::Create::Gadget(mut gadget) => {
                    for connector in gadget.connectors.iter_mut() {
                        connector.gid = gid_vec[(connector.gid as usize) - 1];
                    }
                    let gid = coordinator
                        .execute(bin::Instruction {
                            create: Some(bin::instruction::Create::Gadget(gadget)),
                        })
                        .await
                        .unwrap()
                        .id;
                    gid_vec.push(gid);
                }
                bin::instruction::Create::CheckModel(mut check_model) => {
                    check_model.gid = gid_vec[(check_model.gid as usize) - 1];
                    let cid = coordinator
                        .execute(bin::Instruction {
                            create: Some(bin::instruction::Create::CheckModel(check_model)),
                        })
                        .await
                        .unwrap()
                        .id;
                    cid_vec.push(cid);
                }
                bin::instruction::Create::ErrorModel(mut error_model) => {
                    error_model.cid = cid_vec[(error_model.cid as usize) - 1];
                    let _eid = coordinator
                        .execute(bin::Instruction {
                            create: Some(bin::instruction::Create::ErrorModel(error_model)),
                        })
                        .await
                        .unwrap()
                        .id;
                }
            }
        }
        let mut state = self.state.lock().await;
        state.next_index = 0;
        state.gid_vec = gid_vec;
        state.outcomes.size = 0;
        state.outcomes.data.clear();
        state.loss_mask = None;
        state.decoder_seed = None;
        state.pending_readouts.clear();
    }
}

#[cfg(feature = "cli")]
#[tonic::async_trait]
impl static_controller_server::StaticController for StaticController {
    async fn decode(
        &self,
        request: Request<coordinator::Outcomes>,
    ) -> std::result::Result<Response<coordinator::Readouts>, Status> {
        let batch = self.decode_gadgets(request).await?.into_inner();
        Ok(Response::new(coordinator::Readouts::gather(&batch.readouts)?))
    }

    async fn decode_gadgets(
        &self,
        request: Request<coordinator::Outcomes>,
    ) -> std::result::Result<Response<BatchReadouts>, Status> {
        self.decode_gadgets_through(request, None).await
    }

    async fn reset(&self, _request: Request<()>) -> std::result::Result<Response<()>, Status> {
        let mut coordinator = self.wait_until_library_loaded().await;
        self.reset_and_instantiate_all(&mut coordinator).await;
        Ok(Response::new(()))
    }
}

#[cfg(feature = "cli")]
impl StaticController {
    /// Deliver a batch through an explicit cumulative gadget boundary.
    ///
    /// Timed local callers use this boundary to keep measurement-free gadgets
    /// pending until their own deadlines. `None` preserves measurement-driven RPC behavior.
    pub async fn decode_gadgets_through(
        &self,
        request: Request<coordinator::Outcomes>,
        completed_gadgets: Option<usize>,
    ) -> Result<Response<BatchReadouts>, Status> {
        let mut phases = crate::decoder::worker_trace::WorkerTrace::root(
            "static_batch",
            crate::misc::fastrace::system_diagnostics_enabled(),
        );
        phases.property("completed_gadgets", || format!("{completed_gadgets:?}"));
        let request = request.into_inner();
        let coordinator = self.wait_until_library_loaded().await;
        phases.event("controller_ready");
        if let CoordinatorClient::Local(crate::coordinator::DynCoordinator::CompiledWindow(compiled)) = &coordinator {
            let readouts = compiled.ingest(request, completed_gadgets).await?;
            return Ok(Response::new(BatchReadouts { readouts }));
        }

        let outcomes = request
            .outcomes
            .ok_or_else(|| Status::invalid_argument("Outcomes.outcomes is required"))?;
        crate::misc::validation::validate_outcomes(&outcomes, request.loss_mask.as_ref(), outcomes.size)
            .map_err(Status::invalid_argument)?;
        let loss_bits = request.loss_mask;
        phases.event("input_validated");

        let is_complete;
        {
            // getting the lock to compute the decode requests and spawn tasks
            let mut state = self.state.lock().await;
            phases.event("state_acquired");
            phases.resources("batch_start");
            phases.property("first_gadget", || state.next_index.to_string());
            phases.property("pending_before", || state.pending_decodes.len().to_string());

            let Some(new_size) = state
                .outcomes
                .size
                .checked_add(outcomes.size)
                .and_then(|size| usize::try_from(size).ok())
                .filter(|&size| size <= self.info.total_measurements)
            else {
                return Err(Status::invalid_argument("too many outcomes"));
            };
            if let Some(limit) = completed_gadgets {
                if limit < state.next_index || limit > self.info.accumulated_measurements.len() {
                    return Err(Status::invalid_argument("invalid cumulative gadget boundary"));
                }
                let expected = limit
                    .checked_sub(1)
                    .map_or(0, |index| self.info.accumulated_measurements[index]);
                if new_size != expected {
                    return Err(Status::invalid_argument(
                        "outcomes do not match the cumulative gadget boundary",
                    ));
                }
            }
            let is_first_call_of_shot = state.decoder_seed.is_none();
            crate::coordinator::accept_decoder_seed(&mut state.decoder_seed, request.decoder_seed)?;

            // The Some/None choice for `loss_mask` is locked in by the first
            // decode call of each shot.  Later calls must match that choice.
            if is_first_call_of_shot {
                state.loss_mask = loss_bits;
            } else {
                match (state.loss_mask.as_mut(), loss_bits.as_ref()) {
                    (Some(acc), Some(new)) => bit_vector::append(acc, new),
                    (None, None) => {}
                    (Some(_), None) => {
                        return Err(Status::invalid_argument(
                            "loss_mask was supplied earlier in this shot; subsequent decode \
                             calls of the same shot must also supply it",
                        ));
                    }
                    (None, Some(_)) => {
                        return Err(Status::invalid_argument(
                            "loss_mask was not supplied in the first decode call of this shot; \
                             subsequent calls must not supply it either",
                        ));
                    }
                }
            }
            bit_vector::append(&mut state.outcomes, &outcomes);
            phases.event("outcomes_buffered");

            // send outcomes that are sufficient to decode a gadget
            let mut eager = true;
            let batch_size = self.config.decode_task_batch_size;
            let mut decoding_batch = if batch_size == 1 {
                Vec::new()
            } else {
                Vec::with_capacity(batch_size.min(DECODE_TASK_BATCH_SIZE))
            };
            while state.next_index < completed_gadgets.unwrap_or(self.info.accumulated_measurements.len()) {
                if new_size < self.info.accumulated_measurements[state.next_index] {
                    break; // insufficient outcomes
                }
                let gid = state.gid_vec[state.next_index];
                let start = if state.next_index == 0 {
                    0
                } else {
                    self.info.accumulated_measurements[state.next_index - 1]
                };
                let end = self.info.accumulated_measurements[state.next_index];
                let bit_vector = bit_vector::slice(&state.outcomes, start, end - start);
                let loss_mask_bv = state
                    .loss_mask
                    .as_ref()
                    .map(|mask| bit_vector::slice(mask, start, end - start));
                let dispatch_idx = state.next_index;
                let decoder_seed = state.decoder_seed.flatten();
                state.pending_readouts.push(None);
                state.next_index += 1;

                let coordinator_clone = coordinator.clone();
                let decoding = Box::pin(async move {
                    coordinator_clone
                        .decode(coordinator::Outcomes {
                            gid,
                            outcomes: Some(bit_vector),
                            modifiers: vec![],
                            loss_mask: loss_mask_bv,
                            decoder_seed,
                        })
                        .await
                        .map(|readouts| (dispatch_idx, readouts))
                });
                let mut decoding = crate::misc::poll_trace::TracedPolls::new(decoding, gid);
                // Start one pending request without a waiter-task handoff.
                // Bulk overflow stays asynchronous instead of serializing ingress.
                if let Some(result) = if eager {
                    futures_util::future::poll_immediate(&mut decoding).await
                } else {
                    None
                } {
                    let (index, readouts) = result?;
                    state.pending_readouts[index] = Some(readouts);
                } else {
                    eager = false;
                    if batch_size == 1 {
                        state
                            .pending_decodes
                            .spawn(async move { decoding.await.map(|result| vec![result]) });
                        continue;
                    }
                    decoding_batch.push(decoding);
                    if decoding_batch.len() == batch_size {
                        state
                            .pending_decodes
                            .spawn(futures_util::future::try_join_all(std::mem::take(&mut decoding_batch)));
                    }
                }
            }
            if !decoding_batch.is_empty() {
                state
                    .pending_decodes
                    .spawn(futures_util::future::try_join_all(decoding_batch));
            }
            is_complete = state.next_index == self.info.accumulated_measurements.len();
            phases.event("dispatched");

            // Drain already-completed tasks so the final call has less to join
            while let Some(res) = state.pending_decodes.try_join_next() {
                for (idx, readouts) in res.map_err(|e| Status::internal(format!("join error: {}", e)))?? {
                    state.pending_readouts[idx] = Some(readouts);
                }
            }
            phases.event("drained");
            phases.resources("batch_end");
            phases.property("pending_after", || state.pending_decodes.len().to_string());
        }

        if !is_complete {
            // Partial measurement batch: return empty readouts (tasks are running in background)
            return Ok(Response::new(BatchReadouts::default()));
        }

        // All measurements received: wait for remaining pending decode tasks to complete
        {
            let mut state = self.state.lock().await;
            // First, drain all already-completed tasks without blocking
            while let Some(res) = state.pending_decodes.try_join_next() {
                for (idx, readouts) in res.map_err(|e| Status::internal(format!("join error: {}", e)))?? {
                    state.pending_readouts[idx] = Some(readouts);
                }
            }
            // Then await any still-running tasks (typically just the last spawned one)
            while let Some(res) = state.pending_decodes.join_next().await {
                for (idx, readouts) in res.map_err(|e| Status::internal(format!("join error: {}", e)))?? {
                    state.pending_readouts[idx] = Some(readouts);
                }
            }
        }

        // Gather all readouts in order
        let state = self.state.lock().await;
        let readouts = state
            .pending_readouts
            .iter()
            .cloned()
            .map(|readouts| readouts.ok_or_else(|| Status::internal("missing readouts")))
            .collect::<Result<_, _>>()?;
        Ok(Response::new(BatchReadouts { readouts }))
    }
}

#[cfg(all(test, feature = "cli"))]
#[path = "../../tests/unit/static_controller_test.rs"]
mod tests;
