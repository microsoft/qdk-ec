use crate::bin::{self, check_model, check_model_type, error_model, error_model_type};
use crate::coordinator::CoordinatorClient;
use crate::jit::{self, jit_compiler::JitCompiler};
use crate::misc::sync::TaskCounter;
use hashbrown::{HashMap, HashSet};
use serde::{Deserialize, Serialize};
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
#[cfg(feature = "cli")]
use structdoc::StructDoc;
use tokio::sync::RwLock;
use tokio::sync::oneshot;
use tokio_util::sync::CancellationToken;

include!("../proto/deq.controller.jit_controller.rs");

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "cli", derive(StructDoc))]
#[serde(deny_unknown_fields)]
pub struct JitControllerConfig {
    /// Optional path to a `.deq.jit` library file. When omitted the controller
    /// starts with an empty library and the caller is expected to load
    /// libraries dynamically via the `load_library` RPC (this is the typical
    /// path for in-process Python use).
    #[serde(default)]
    pub filepath: Option<String>,
    /// cache the generated deq-bin and try to reuse them instead of creating new ones.
    /// this will help the coordinator to find cached decoder results and speed up decoding, but
    /// may cause memory bloat if there are many unique check models and error models.
    #[serde(default = "default_cache_enabled")]
    pub cache_enabled: bool,
}

fn default_cache_enabled() -> bool {
    true
}

pub struct JitController {
    pub config: JitControllerConfig,
    pub compiler: Arc<JitCompiler>,
    coordinator: RwLock<Option<CoordinatorClient>>,
    type_cache: RwLock<TypeCache>,
    next_ctype: AtomicU64,
    next_etype: AtomicU64,
    library: crate::jit::JitLibrary,
    /// Track when error models are loaded for each gid.
    /// Decode must wait for the error model before forwarding to coordinator.
    /// Stores the receiver; the sender is passed to the spawned error model task.
    error_model_loaded: RwLock<HashMap<u64, oneshot::Receiver<()>>>,
    /// Cancelled on reset()/drop to abort pending error-model and batch tasks.
    cancellation: RwLock<CancellationToken>,
    /// Tracks active spawned tasks; reset() waits for all to finish.
    task_counter: Arc<TaskCounter>,
}

impl JitController {
    pub fn new(config: serde_json::Value) -> Arc<Self> {
        let config: JitControllerConfig = serde_json::from_value(config).unwrap();
        let library: crate::jit::JitLibrary = if let Some(filepath) = config.filepath.as_ref() {
            let data = std::fs::read(filepath).unwrap();
            prost::Message::decode(&mut data.as_slice()).unwrap()
        } else {
            crate::jit::JitLibrary::default()
        };
        let compiler = JitCompiler::new();
        Arc::new(Self {
            config,
            compiler,
            coordinator: RwLock::new(None),
            type_cache: RwLock::new(TypeCache::new()),
            next_ctype: AtomicU64::new(1),
            next_etype: AtomicU64::new(1),
            library,
            error_model_loaded: RwLock::new(HashMap::new()),
            cancellation: RwLock::new(CancellationToken::new()),
            task_counter: TaskCounter::new(),
        })
    }

    /// Create a JitController from an in-memory library.
    pub fn new_from_library(library: crate::jit::JitLibrary, cache_enabled: bool) -> Arc<Self> {
        let compiler = JitCompiler::new();
        Arc::new(Self {
            config: JitControllerConfig {
                filepath: None,
                cache_enabled,
            },
            compiler,
            coordinator: RwLock::new(None),
            type_cache: RwLock::new(TypeCache::new()),
            next_ctype: AtomicU64::new(1),
            next_etype: AtomicU64::new(1),
            library,
            error_model_loaded: RwLock::new(HashMap::new()),
            cancellation: RwLock::new(CancellationToken::new()),
            task_counter: TaskCounter::new(),
        })
    }
    pub async fn start(self: &Arc<Self>, client: CoordinatorClient) {
        {
            let mut coordinator = self.coordinator.write().await;
            coordinator.replace(client);
        }
        self.load_library(self.library.clone())
            .await
            .expect("failed to load initial library at start");
    }

    /// Load a JIT library: register types with the JIT compiler AND forward
    /// the underlying `bin` port/gadget types to the coordinator so that
    /// subsequent `execute` calls can reference them. Used at start-time for
    /// the initial library and at runtime for dynamically-loaded libraries
    /// (e.g. from Python). Loading is additive — repeated calls accumulate.
    pub async fn load_library(self: &Arc<Self>, library: jit::JitLibrary) -> Result<(), tonic::Status> {
        let port_types: Vec<_> = library.port_types.iter().filter_map(|pt| pt.base.clone()).collect();
        let gadget_types: Vec<_> = library.gadget_types.iter().filter_map(|gt| gt.base.clone()).collect();

        self.compiler.load_library(library).await;

        if port_types.is_empty() && gadget_types.is_empty() {
            return Ok(());
        }
        let coordinator_guard = self.coordinator.read().await;
        let coordinator = coordinator_guard
            .as_ref()
            .ok_or_else(|| tonic::Status::failed_precondition("coordinator not connected"))?;
        coordinator
            .load_library(bin::Library {
                port_types,
                gadget_types,
                ..Default::default()
            })
            .await
    }

    pub fn next_ctype(&self) -> u64 {
        self.next_ctype.fetch_add(1, Ordering::SeqCst)
    }

    pub fn next_etype(&self) -> u64 {
        self.next_etype.fetch_add(1, Ordering::SeqCst)
    }

    /// Get cached ctype or allocate and load a new one atomically.
    /// Holds the lock until the type is loaded to coordinator to prevent races.
    async fn get_or_load_ctype(&self, check_model_type: &mut bin::CheckModelType, coordinator: &CoordinatorClient) -> u64 {
        let mut cache = self.type_cache.write().await;
        if self.config.cache_enabled
            && let Some(cached) = cache.get_check_model_type(check_model_type)
        {
            return cached;
        }
        let ctype = self.next_ctype();
        check_model_type.ctype = ctype;
        coordinator
            .load_library(bin::Library {
                check_model_types: vec![check_model_type.clone()],
                ..Default::default()
            })
            .await
            .unwrap();
        if self.config.cache_enabled {
            cache.insert_check_model_type(check_model_type, ctype);
        }
        ctype
    }

    /// Get cached etype or allocate and load a new one atomically.
    /// Holds the lock until the type is loaded to coordinator to prevent races.
    async fn get_or_load_etype(&self, error_model_type: &mut bin::ErrorModelType, coordinator: &CoordinatorClient) -> u64 {
        let mut cache = self.type_cache.write().await;
        if self.config.cache_enabled
            && let Some(cached) = cache.get_error_model_type(error_model_type)
        {
            return cached;
        }
        let etype = self.next_etype();
        error_model_type.etype = etype;
        coordinator
            .load_library(bin::Library {
                error_model_types: vec![error_model_type.clone()],
                ..Default::default()
            })
            .await
            .unwrap();
        if self.config.cache_enabled {
            cache.insert_error_model_type(error_model_type, etype);
        }
        etype
    }

    pub async fn clear_cache(&self) {
        self.type_cache.write().await.clear();
    }

    /// Execute a JIT instruction, compiling it and sending commands to the coordinator.
    ///
    /// Returns the assigned gid for the instantiated gadget once the gadget and check model
    /// are loaded. The error model is loaded asynchronously in the background to avoid
    /// circular dependencies (error models depend on future gadgets' gids).
    pub async fn execute(self: &Arc<Self>, instruction: jit::JitInstruction) -> u64 {
        let token = self.cancellation.read().await.clone();
        let (gadget, mut check_model_type, check_model, error_model_future) =
            Arc::clone(&self.compiler).compile(instruction, token.clone()).await;

        let gid = gadget.gid;
        let cid = gid;

        // Create a oneshot channel to track when the error model is loaded
        let (error_model_tx, error_model_rx) = oneshot::channel();
        self.error_model_loaded.write().await.insert(gid, error_model_rx);

        {
            let coordinator_guard = self.coordinator.read().await;
            let coordinator = coordinator_guard.as_ref().expect("coordinator not connected");

            // NOTE: We assume the coordinator returns the same gid we sent. This assumption
            // holds for single-node coordinators but may break in distributed decoding systems
            // where a global id assignment scheme would require the coordinator to return
            // a different (globally unique) id. In that case, we would need to use the returned
            // id and update all subsequent references (cid, check_model, error_model, etc.).
            let returned_gid = coordinator
                .execute(bin::Instruction {
                    create: Some(bin::instruction::Create::Gadget(gadget)),
                })
                .await
                .unwrap()
                .id;
            assert_eq!(
                returned_gid, gid,
                "coordinator returned different gid; distributed id assignment not yet supported"
            );

            let ctype = self.get_or_load_ctype(&mut check_model_type, coordinator).await;

            let check_model_modifier = build_check_model_modifier(&check_model_type);
            coordinator
                .execute(bin::Instruction {
                    create: Some(bin::instruction::Create::CheckModel(bin::CheckModel {
                        ctype,
                        gid,
                        tag: check_model.tag.clone(),
                        modifier: check_model_modifier,
                        cid: check_model.cid,
                    })),
                })
                .await
                .unwrap();
        }

        // Spawn error model loading in background. The JIT compiler determines when the
        // error model future resolves (once output ports are connected), and we send it
        // to the coordinator as soon as it's ready.
        let this = Arc::clone(self);
        let _guard = self.task_counter.guard();
        tokio::spawn(async move {
            let _guard = _guard;
            let (mut error_model_type, error_model) = tokio::select! {
                result = error_model_future => result,
                _ = token.cancelled() => { return; }
            };

            let coordinator_guard = this.coordinator.read().await;
            let Some(coordinator) = coordinator_guard.as_ref() else {
                return;
            };

            let etype = this.get_or_load_etype(&mut error_model_type, coordinator).await;

            let error_model_modifier = build_error_model_modifier(&error_model_type, &error_model);
            let _ = coordinator
                .execute(bin::Instruction {
                    create: Some(bin::instruction::Create::ErrorModel(bin::ErrorModel {
                        etype,
                        cid,
                        tag: error_model.tag.clone(),
                        modifier: error_model_modifier,
                        eid: error_model.eid,
                    })),
                })
                .await;

            // Notify that the error model has been loaded
            let _ = error_model_tx.send(());
        });

        gid
    }

    /// Execute multiple JIT instructions in batch, respecting dependencies between them.
    ///
    /// All instructions must specify a non-zero gid. Dependencies (via connectors) are validated:
    /// referenced gids must either exist in this batch or have been previously executed.
    /// The entire batch is rejected if any validation fails.
    ///
    /// Returns the gids in the same order as the input instructions.
    ///
    /// # Errors
    ///
    /// Returns `BatchExecuteError` if validation fails.
    ///
    /// # Panics
    ///
    /// Panics if any spawned task panics during execution.
    pub async fn batch_execute(
        self: &Arc<Self>,
        instructions: Vec<jit::JitInstruction>,
    ) -> Result<Vec<u64>, BatchExecuteError> {
        let mut batch_gids: HashSet<u64> = HashSet::new();
        let mut gid_to_index: HashMap<u64, usize> = HashMap::new();

        for (index, instruction) in instructions.iter().enumerate() {
            let gadget = instruction
                .gadget
                .as_ref()
                .ok_or(BatchExecuteError::MissingGadget { index })?;
            let gid = gadget.gid;
            if gid == 0 {
                return Err(BatchExecuteError::ZeroGid { index });
            }
            if batch_gids.contains(&gid) {
                return Err(BatchExecuteError::DuplicateGid { gid, index });
            }
            batch_gids.insert(gid);
            gid_to_index.insert(gid, index);
        }

        for (index, instruction) in instructions.iter().enumerate() {
            let gadget = instruction.gadget.as_ref().unwrap();
            for connector in &gadget.connectors {
                let dep_gid = connector.gid;
                if !batch_gids.contains(&dep_gid) && !self.compiler.contains_gid(dep_gid).await {
                    return Err(BatchExecuteError::MissingDependency {
                        index,
                        gid: gadget.gid,
                        missing_gid: dep_gid,
                    });
                }
            }
        }

        let mut dependencies: Vec<Vec<usize>> = vec![Vec::new(); instructions.len()];
        for (index, instruction) in instructions.iter().enumerate() {
            let gadget = instruction.gadget.as_ref().unwrap();
            for connector in &gadget.connectors {
                let dep_gid = connector.gid;
                if let Some(&dep_index) = gid_to_index.get(&dep_gid) {
                    dependencies[index].push(dep_index);
                }
            }
        }

        let gids: Vec<u64> = instructions.iter().map(|i| i.gadget.as_ref().unwrap().gid).collect();
        let num_instructions = instructions.len();
        let instructions = Arc::new(instructions);
        // Use watch channels so that late subscribers still see the completion
        // signal (broadcast channels drop messages if no subscriber exists yet).
        let completion_txs: Arc<Vec<_>> =
            Arc::new((0..num_instructions).map(|_| tokio::sync::watch::channel(false).0).collect());

        let mut handles = Vec::with_capacity(num_instructions);
        let token = self.cancellation.read().await.clone();
        for index in 0..num_instructions {
            let this = Arc::clone(self);
            let instructions = Arc::clone(&instructions);
            let deps = dependencies[index].clone();
            let completion_txs = Arc::clone(&completion_txs);
            let token = token.clone();

            let handle = tokio::spawn(async move {
                for &dep_index in &deps {
                    let mut rx = completion_txs[dep_index].subscribe();
                    tokio::select! {
                        _ = rx.wait_for(|&done| done) => {}
                        _ = token.cancelled() => { return None; }
                    }
                }

                let instruction = instructions[index].clone();
                let gid = this.execute(instruction).await;

                completion_txs[index].send_replace(true);
                Some(gid)
            });
            handles.push(handle);
        }

        let results: Vec<u64> = futures_util::future::join_all(handles)
            .await
            .into_iter()
            .map(|r| r.expect("batch execute task panicked"))
            .collect::<Option<Vec<_>>>()
            .unwrap_or(gids.clone());

        debug_assert_eq!(results, gids);
        Ok(gids)
    }

    /// Decode multiple gadgets concurrently.
    ///
    /// All decode operations run in parallel, which is critical for window decoding
    /// where earlier decodes may depend on later ones. Returns readouts in the same
    /// order as the input outcomes.
    ///
    /// # Errors
    ///
    /// Returns `tonic::Status` if any decode operation fails.
    ///
    /// # Panics
    ///
    /// Panics if any spawned task panics during decoding.
    pub async fn batch_decode(
        self: &Arc<Self>,
        outcomes: Vec<crate::coordinator::Outcomes>,
    ) -> Result<Vec<crate::coordinator::Readouts>, tonic::Status> {
        let this = Arc::clone(self);
        let token = self.cancellation.read().await.clone();
        let handles: Vec<_> = outcomes
            .into_iter()
            .map(|outcome| {
                let this = Arc::clone(&this);
                let token = token.clone();
                tokio::spawn(async move {
                    tokio::select! {
                        r = this.decode_single(outcome) => Some(r),
                        _ = token.cancelled() => None,
                    }
                })
            })
            .collect();

        let mut results = Vec::with_capacity(handles.len());
        for r in futures_util::future::join_all(handles).await {
            match r.expect("batch decode task panicked") {
                Some(Ok(readouts)) => results.push(readouts),
                Some(Err(e)) => return Err(e),
                None => {
                    return Err(tonic::Status::cancelled("batch decode cancelled by reset"));
                }
            }
        }
        Ok(results)
    }

    pub async fn decode_single(
        self: &Arc<Self>,
        outcomes: crate::coordinator::Outcomes,
    ) -> Result<crate::coordinator::Readouts, tonic::Status> {
        let gid = outcomes.gid;

        let rx = self.error_model_loaded.write().await.remove(&gid).ok_or_else(|| {
            tonic::Status::invalid_argument(format!("decode called for unknown or already-decoded gid: {gid}"))
        })?;
        // Wait for the background error-model loading task to complete OR for
        // the cancellation token to fire (e.g. on runtime shutdown). The
        // oneshot resolves with Err(RecvError) if the sender is dropped, so
        // it cannot hang on its own — but the upstream error-model future may
        // itself be waiting on an unconnected output port that will never
        // come. Selecting on the token guarantees we surface a cancellation
        // promptly rather than blocking forever.
        let token = self.cancellation.read().await.clone();
        tokio::select! {
            _ = rx => {}
            _ = token.cancelled() => {
                return Err(tonic::Status::cancelled(format!(
                    "decode for gid={gid} cancelled by runtime shutdown or reset"
                )));
            }
        }

        let coordinator_guard = self.coordinator.read().await;
        let coordinator = coordinator_guard
            .as_ref()
            .ok_or_else(|| tonic::Status::failed_precondition("coordinator not connected"))?;
        coordinator.decode(outcomes).await
    }

    /// Fire the cancellation token to abort any pending error-model loads and
    /// in-flight decodes. Used by [`crate::server::LocalServer::shutdown`] to
    /// propagate runtime shutdown into the service layer. Unlike [`Self::reset`],
    /// this does not wait for tasks to finish, does not refresh the token, and
    /// does not clear caches — it just signals every cancellable point to bail.
    pub async fn cancel_pending(&self) {
        let token = self.cancellation.read().await;
        token.cancel();
    }

    pub async fn reset(self: &Arc<Self>, mut flags: crate::coordinator::ResetRequest) -> Result<(), tonic::Status> {
        // Resetting the library invalidates decoder caches that reference
        // library-assigned type IDs, so the decoder service must be reset too.
        if flags.reset_library {
            flags.reset_decoder_service = true;
        }
        // Cancel all pending async tasks, wait for them to finish, then
        // install a fresh token so post-reset operations proceed normally.
        {
            let token = self.cancellation.read().await;
            token.cancel();
        }
        self.task_counter.wait_for_zero().await;
        {
            let mut token = self.cancellation.write().await;
            *token = CancellationToken::new();
        }
        let reset_library = flags.reset_library;
        self.compiler.reset().await;
        if reset_library {
            self.clear_cache().await;
        }
        self.error_model_loaded.write().await.clear();
        let coordinator_guard = self.coordinator.read().await;
        if let Some(coordinator) = coordinator_guard.as_ref() {
            coordinator.reset(flags).await?;

            // Re-send port types and gadget types so the coordinator stays consistent
            // with what the JIT controller expects to reference in future compiles.
            if reset_library {
                self.next_ctype.store(1, Ordering::SeqCst);
                self.next_etype.store(1, Ordering::SeqCst);
                let port_types: Vec<_> = self.library.port_types.iter().map(|pt| pt.base.clone().unwrap()).collect();
                let gadget_types: Vec<_> = self.library.gadget_types.iter().map(|gt| gt.base.clone().unwrap()).collect();
                coordinator
                    .load_library(bin::Library {
                        port_types,
                        gadget_types,
                        ..Default::default()
                    })
                    .await
                    .unwrap();
            }
        }
        Ok(())
    }

    #[cfg(feature = "cli")]
    pub fn add_service(self: &Arc<Self>, router: tonic::transport::server::Router) -> tonic::transport::server::Router {
        let service = jit_controller_server::JitControllerServer::new(JitControllerService(Arc::clone(self)))
            .max_decoding_message_size(usize::MAX);
        router.add_service(service)
    }
}

#[derive(Debug, Clone)]
pub enum BatchExecuteError {
    MissingGadget { index: usize },
    ZeroGid { index: usize },
    DuplicateGid { gid: u64, index: usize },
    MissingDependency { index: usize, gid: u64, missing_gid: u64 },
}

impl std::fmt::Display for BatchExecuteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingGadget { index } => {
                write!(f, "instruction at index {index} has no gadget")
            }
            Self::ZeroGid { index } => {
                write!(
                    f,
                    "instruction at index {index} has gid=0; all batch instructions must specify non-zero gid"
                )
            }
            Self::DuplicateGid { gid, index } => {
                write!(f, "instruction at index {index} has duplicate gid={gid}")
            }
            Self::MissingDependency { index, gid, missing_gid } => {
                write!(
                    f,
                    "instruction at index {index} (gid={gid}) depends on gid={missing_gid} which does not exist"
                )
            }
        }
    }
}

impl std::error::Error for BatchExecuteError {}

#[cfg(feature = "cli")]
impl From<BatchExecuteError> for tonic::Status {
    fn from(err: BatchExecuteError) -> Self {
        tonic::Status::invalid_argument(err.to_string())
    }
}

/// Wrapper that holds Arc<JitController> for implementing the gRPC trait.
/// This allows gRPC methods to call Arc-based methods like `execute`.
#[cfg(feature = "cli")]
struct JitControllerService(Arc<JitController>);

#[cfg(feature = "cli")]
#[tonic::async_trait]
impl jit_controller_server::JitController for JitControllerService {
    async fn load_library(&self, request: tonic::Request<jit::JitLibrary>) -> Result<tonic::Response<()>, tonic::Status> {
        let library = request.into_inner();
        self.0.load_library(library).await?;
        Ok(tonic::Response::new(()))
    }

    async fn unload(&self, _request: tonic::Request<jit::UnloadJitLibrary>) -> Result<tonic::Response<()>, tonic::Status> {
        Err(tonic::Status::unimplemented("unload not implemented"))
    }

    async fn execute(
        &self,
        request: tonic::Request<jit::JitInstruction>,
    ) -> Result<tonic::Response<crate::coordinator::ExecuteResponse>, tonic::Status> {
        let instruction = request.into_inner();
        let gid = self.0.execute(instruction).await;
        Ok(tonic::Response::new(crate::coordinator::ExecuteResponse { id: gid }))
    }

    async fn batch_execute(
        &self,
        request: tonic::Request<BatchExecuteRequest>,
    ) -> Result<tonic::Response<BatchExecuteResponse>, tonic::Status> {
        let batch_request = request.into_inner();
        let gids = self.0.batch_execute(batch_request.instructions).await?;
        Ok(tonic::Response::new(BatchExecuteResponse { gids }))
    }

    async fn decode(
        &self,
        request: tonic::Request<crate::coordinator::Outcomes>,
    ) -> Result<tonic::Response<crate::coordinator::Readouts>, tonic::Status> {
        let outcomes = request.into_inner();
        let gid = outcomes.gid;

        // Wait for the error model to be loaded before forwarding to coordinator.
        // This is necessary because JIT compilation loads error models asynchronously.
        // We remove the entry since each gid should only be decoded once.
        let rx = self
            .0
            .error_model_loaded
            .write()
            .await
            .remove(&gid)
            .expect("decode called for unknown or already-decoded gid");
        let _ = rx.await;

        let coordinator_guard = self.0.coordinator.read().await;
        let coordinator = coordinator_guard
            .as_ref()
            .ok_or_else(|| tonic::Status::failed_precondition("coordinator not connected"))?;
        let readouts = coordinator.decode(outcomes).await?;
        Ok(tonic::Response::new(readouts))
    }

    async fn batch_decode(
        &self,
        request: tonic::Request<BatchOutcomes>,
    ) -> Result<tonic::Response<BatchReadouts>, tonic::Status> {
        let batch_outcomes = request.into_inner();
        let readouts = self.0.batch_decode(batch_outcomes.outcomes).await?;
        Ok(tonic::Response::new(BatchReadouts { readouts }))
    }

    async fn reset(
        &self,
        request: tonic::Request<crate::coordinator::ResetRequest>,
    ) -> Result<tonic::Response<()>, tonic::Status> {
        let flags = request.into_inner();
        self.0.reset(flags).await?;
        Ok(tonic::Response::new(()))
    }
}

/// Build check model modifier that rewrites all remote gadgets with absolute indices.
/// JIT compiler outputs absolute gid values which must be passed via modifier.
fn build_check_model_modifier(check_model_type: &bin::CheckModelType) -> Option<check_model::CheckModelModifier> {
    if check_model_type.remote_gadgets.is_empty() {
        return None;
    }

    let reroutes: Vec<_> = check_model_type
        .remote_gadgets
        .iter()
        .enumerate()
        .map(
            |(idx, remote_gadget)| check_model::check_model_modifier::RerouteRemoteGadget {
                remote_gadget_index: idx as u64,
                value: Some(remote_gadget.clone()),
            },
        )
        .collect();

    Some(check_model::CheckModelModifier {
        reroute_remote_gadgets: reroutes,
    })
}

/// Build error model modifier that rewrites all remote check models with absolute indices
/// and includes the probability modifier from the error model.
fn build_error_model_modifier(
    error_model_type: &bin::ErrorModelType,
    error_model: &bin::ErrorModel,
) -> Option<error_model::ErrorModelModifier> {
    let reroutes: Vec<_> = error_model_type
        .remote_check_models
        .iter()
        .enumerate()
        .map(
            |(idx, remote_check_model)| error_model::error_model_modifier::RerouteRemoteCheckModel {
                remote_check_model_index: idx as u64,
                value: Some(remote_check_model.clone()),
            },
        )
        .collect();

    let probability_modifier = error_model.modifier.as_ref().and_then(|m| m.probability_modifier.clone());

    let has_reroutes = !reroutes.is_empty();
    let has_probability = probability_modifier.is_some();

    if !has_reroutes && !has_probability {
        None
    } else {
        Some(error_model::ErrorModelModifier {
            probability_modifier,
            reroute_remote_check_models: reroutes,
        })
    }
}

/// Wrapper for CheckModelType that implements Hash/Eq based on structural semantics only.
/// Excludes: ctype (ID), name, description, and visualization fields (tag, relative).
#[derive(Clone)]
pub struct CheckModelTypeKey(pub bin::CheckModelType);

impl PartialEq for CheckModelTypeKey {
    fn eq(&self, other: &Self) -> bool {
        let a = &self.0;
        let b = &other.0;
        a.gtype == b.gtype
            && a.checks.len() == b.checks.len()
            && a.checks.iter().zip(b.checks.iter()).all(|(ca, cb)| checks_eq(ca, cb))
    }
}

impl Eq for CheckModelTypeKey {}

impl Hash for CheckModelTypeKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let cmt = &self.0;
        cmt.gtype.hash(state);
        cmt.checks.len().hash(state);
        for check in &cmt.checks {
            hash_check(check, state);
        }
    }
}

impl std::fmt::Debug for CheckModelTypeKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CheckModelTypeKey")
            .field("gtype", &self.0.gtype)
            .field("checks_len", &self.0.checks.len())
            .finish()
    }
}

/// Wrapper for ErrorModelType that implements Hash/Eq based on structural semantics only.
/// Excludes: etype (ID), name, description, and visualization fields (tag, relative).
#[derive(Clone)]
pub struct ErrorModelTypeKey(pub bin::ErrorModelType);

impl PartialEq for ErrorModelTypeKey {
    fn eq(&self, other: &Self) -> bool {
        let a = &self.0;
        let b = &other.0;
        a.ctype == b.ctype
            && a.errors.len() == b.errors.len()
            && a.errors.iter().zip(b.errors.iter()).all(|(ea, eb)| errors_eq(ea, eb))
    }
}

impl Eq for ErrorModelTypeKey {}

impl Hash for ErrorModelTypeKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        hash_error_model_type_structural(&self.0, state);
    }
}

/// Structural hash for [`bin::ErrorModelType`] — the same fields hashed by
/// [`ErrorModelTypeKey::hash`], exposed as a free function so callers can
/// fingerprint a borrowed `ErrorModelType` without constructing (and cloning
/// into) an [`ErrorModelTypeKey`].
///
/// Used by `coordinator::decoder_cache_key::etype_digest` to fold the resolved
/// error-model-type structure (including the full `errors` list) into the
/// `loaded_decoders` cache key.
pub(crate) fn hash_error_model_type_structural<H: Hasher>(emt: &bin::ErrorModelType, state: &mut H) {
    emt.ctype.hash(state);
    emt.errors.len().hash(state);
    for error in &emt.errors {
        hash_error(error, state);
    }
}

impl std::fmt::Debug for ErrorModelTypeKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ErrorModelTypeKey")
            .field("ctype", &self.0.ctype)
            .field("errors_len", &self.0.errors.len())
            .finish()
    }
}

fn checks_eq(a: &check_model_type::Check, b: &check_model_type::Check) -> bool {
    a.measurements == b.measurements && a.naturally_flipped == b.naturally_flipped
}

fn hash_check<H: Hasher>(check: &check_model_type::Check, state: &mut H) {
    check.measurements.hash(state);
    check.naturally_flipped.hash(state);
}

fn errors_eq(a: &error_model_type::Error, b: &error_model_type::Error) -> bool {
    a.checks == b.checks
        && a.residual == b.residual
        && a.readout_flips == b.readout_flips
        && a.probability.to_bits() == b.probability.to_bits()
}

fn hash_error<H: Hasher>(error: &error_model_type::Error, state: &mut H) {
    error.checks.hash(state);
    error.residual.hash(state);
    error.readout_flips.hash(state);
    error.probability.to_bits().hash(state);
}

/// Cache for reusing CheckModelType and ErrorModelType across JIT compilations.
#[derive(Default)]
pub struct TypeCache {
    check_model_types: HashMap<CheckModelTypeKey, u64>,
    error_model_types: HashMap<ErrorModelTypeKey, u64>,
}

impl TypeCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get_check_model_type(&self, check_model_type: &bin::CheckModelType) -> Option<u64> {
        let key = CheckModelTypeKey(check_model_type.clone());
        self.check_model_types.get(&key).copied()
    }

    pub fn insert_check_model_type(&mut self, check_model_type: &bin::CheckModelType, ctype: u64) {
        let key = CheckModelTypeKey(check_model_type.clone());
        self.check_model_types.insert(key, ctype);
    }

    pub fn get_error_model_type(&self, error_model_type: &bin::ErrorModelType) -> Option<u64> {
        let key = ErrorModelTypeKey(error_model_type.clone());
        self.error_model_types.get(&key).copied()
    }

    pub fn insert_error_model_type(&mut self, error_model_type: &bin::ErrorModelType, etype: u64) {
        let key = ErrorModelTypeKey(error_model_type.clone());
        self.error_model_types.insert(key, etype);
    }

    pub fn clear(&mut self) {
        self.check_model_types.clear();
        self.error_model_types.clear();
    }
}
