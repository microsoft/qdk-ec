use crate::bin;
use crate::coordinator;
use crate::coordinator::{CoordinatorClient, ResetRequest};
#[cfg(feature = "cli")]
use crate::util::BitVector;
use serde::{Deserialize, Serialize};
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
    outcomes: Vec<bool>,
    /// background decode tasks for streaming mode (dispatched but not yet awaited)
    pending_decodes: JoinSet<Result<(usize, coordinator::Readouts), Status>>,
    /// readouts collected from completed background decode tasks
    pending_readouts: Vec<Option<coordinator::Readouts>>,
    /// index offset for pending readouts (how many gadgets have been dispatched so far)
    dispatched_count: usize,
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
        let config: StaticControllerConfig = serde_json::from_value(config).unwrap();
        // read from file
        let data = std::fs::read(config.filepath.clone()).unwrap();
        let library: bin::Library = prost::Message::decode(&mut data.as_slice()).unwrap();
        let info = Arc::new(StaticControllerInfo::new(&library));
        let state = StaticControllerState {
            next_index: 0,
            gid_vec: Vec::with_capacity(info.accumulated_measurements.len()),
            outcomes: Vec::with_capacity(info.total_measurements),
            pending_decodes: JoinSet::new(),
            pending_readouts: Vec::new(),
            dispatched_count: 0,
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
        // since a static program should never change, we reset the coordinator to a minimum extent
        coordinator
            .reset(ResetRequest {
                reset_library: self.config.reset_library,
                reset_decoder_service: self.config.reset_decoder_service,
                ..Default::default()
            })
            .await
            .unwrap();
        let mut gid_vec: Vec<u64> = vec![];
        let mut cid_vec: Vec<u64> = vec![];
        for instruction in self.library.program.iter() {
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
        state.outcomes.clear();
        state.pending_decodes.shutdown().await;
        state.pending_readouts.clear();
        state.dispatched_count = 0;
    }
}

#[cfg(feature = "cli")]
#[tonic::async_trait]
impl static_controller_server::StaticController for StaticController {
    async fn decode(&self, request: Request<BitVector>) -> std::result::Result<Response<BitVector>, Status> {
        let outcomes = request.into_inner();
        let coordinator = self.wait_until_library_loaded().await;

        let outcomes = crate::misc::bit_vector::unpack_bits(&outcomes.data, outcomes.size);
        let is_complete;
        {
            // getting the lock to compute the decode requests and spawn tasks
            let mut state = self.state.lock().await;
            state.outcomes.extend_from_slice(&outcomes);
            assert!(state.outcomes.len() <= self.info.total_measurements, "too many outcomes");
            is_complete = state.outcomes.len() == self.info.total_measurements;

            // send outcomes that are sufficient to decode a gadget
            while state.next_index < self.info.accumulated_measurements.len() {
                if state.outcomes.len() < self.info.accumulated_measurements[state.next_index] {
                    break; // insufficient outcomes
                }
                let gid = state.gid_vec[state.next_index];
                let start = if state.next_index == 0 {
                    0
                } else {
                    self.info.accumulated_measurements[state.next_index - 1]
                };
                let slice: Vec<bool> = state.outcomes[start..self.info.accumulated_measurements[state.next_index]].into();
                let bit_vector = BitVector {
                    size: slice.len() as u64,
                    data: crate::misc::bit_vector::pack_bits(&slice),
                };
                let dispatch_idx = state.dispatched_count;
                state.dispatched_count += 1;
                state.pending_readouts.push(None);

                let coordinator_clone = coordinator.clone();
                state.pending_decodes.spawn(async move {
                    coordinator_clone
                        .decode(coordinator::Outcomes {
                            gid,
                            outcomes: Some(bit_vector),
                            modifiers: vec![],
                        })
                        .await
                        .map(|readouts| (dispatch_idx, readouts))
                });
                state.next_index += 1;
            }

            // Drain already-completed tasks so the final call has less to join
            while let Some(res) = state.pending_decodes.try_join_next() {
                let (idx, readouts) = res.map_err(|e| Status::internal(format!("join error: {}", e)))??;
                state.pending_readouts[idx] = Some(readouts);
            }
        }

        if !is_complete {
            // Partial measurement batch: return empty readouts (tasks are running in background)
            let empty = BitVector { size: 0, data: vec![] };
            return Ok(empty.into());
        }

        // All measurements received: wait for remaining pending decode tasks to complete
        {
            let mut state = self.state.lock().await;
            // First, drain all already-completed tasks without blocking
            while let Some(res) = state.pending_decodes.try_join_next() {
                let (idx, readouts) = res.map_err(|e| Status::internal(format!("join error: {}", e)))??;
                state.pending_readouts[idx] = Some(readouts);
            }
            // Then await any still-running tasks (typically just the last spawned one)
            while let Some(res) = state.pending_decodes.join_next().await {
                let (idx, readouts) = res.map_err(|e| Status::internal(format!("join error: {}", e)))??;
                state.pending_readouts[idx] = Some(readouts);
            }
        }

        // Gather all readouts in order
        let state = self.state.lock().await;
        let mut gathered_readouts = vec![];
        for readouts in state.pending_readouts.iter() {
            if let Some(r) = readouts {
                let bit_vector = r
                    .readouts
                    .as_ref()
                    .ok_or_else(|| Status::internal("empty bit vector in readouts"))?;
                gathered_readouts
                    .extend_from_slice(&crate::misc::bit_vector::unpack_bits(&bit_vector.data, bit_vector.size));
            } else {
                return Err(Status::internal("missing readouts"));
            }
        }

        let gathered_readouts = BitVector {
            size: gathered_readouts.len() as u64,
            data: crate::misc::bit_vector::pack_bits(&gathered_readouts),
        };
        Ok(gathered_readouts.into())
    }

    async fn reset(&self, _request: Request<()>) -> std::result::Result<Response<()>, Status> {
        let mut coordinator = self.wait_until_library_loaded().await;
        self.reset_and_instantiate_all(&mut coordinator).await;
        Ok(Response::new(()))
    }
}
