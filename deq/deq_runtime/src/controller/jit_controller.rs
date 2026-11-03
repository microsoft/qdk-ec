use crate::bin::{self, check_model, check_model_type, error_model, error_model_type};
use crate::coordinator::CoordinatorClient;
use crate::jit::{self, jit_compiler::JitCompiler};
use crate::misc::sync::TaskCounter;
use hashbrown::{HashMap, HashSet};
use serde::{Deserialize, Serialize};
use std::hash::{BuildHasher, Hash, Hasher};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Weak};
#[cfg(feature = "cli")]
use structdoc::StructDoc;
use tokio::sync::{Mutex, RwLock};
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
    /// Positive integer multiplier of registered JIT gadget types, default 3.
    /// Sets the entry budget for each generated check/error type cache and the
    /// error-type identity cache, not a per-gadget quota. Caches use LRU eviction and survive shot resets
    /// independently of decoder persistence. Live declarations can outlast their cached entries.
    #[serde(default = "default_cache_limit_ratio", deserialize_with = "deserialize_cache_limit_ratio")]
    pub cache_limit_ratio: usize,
}

impl JitControllerConfig {
    fn cache_limit(&self, gadget_types: usize) -> usize {
        gadget_types.saturating_mul(self.cache_limit_ratio).max(1)
    }
}

fn default_cache_limit_ratio() -> usize {
    3
}

fn deserialize_cache_limit_ratio<'de, D: serde::Deserializer<'de>>(deserializer: D) -> Result<usize, D::Error> {
    std::num::NonZeroUsize::deserialize(deserializer).map(std::num::NonZeroUsize::get)
}

pub struct JitController {
    pub config: JitControllerConfig,
    pub compiler: Arc<JitCompiler>,
    coordinator: RwLock<Option<CoordinatorClient>>,
    type_cache: RwLock<TypeCache>,
    next_ctype: AtomicU64,
    next_etype: AtomicU64,
    terminal_etypes: RwLock<HashMap<u64, u64>>,
    library: crate::jit::JitLibrary,
    /// Executed gadgets whose outcomes have not yet been submitted for decoding.
    undecoded_gids: RwLock<HashSet<u64>>,
    /// Cancelled on reset()/drop to abort pending error-model and batch tasks.
    cancellation: RwLock<CancellationToken>,
    /// First background failure in this shot; stored before cancellation wakes callers.
    background_error: std::sync::Mutex<Option<tonic::Status>>,
    /// Tracks active spawned tasks; reset() waits for all to finish.
    task_counter: Arc<TaskCounter>,
    /// Publication barrier: executions hold readers; generated-type unloading takes the writer.
    collection: Arc<RwLock<()>>,
    /// Request generation used to coalesce concurrent garbage-collection requests.
    collection_requests: AtomicU64,
    /// Last successfully collected generation; the mutex serializes collection passes.
    collection_completed: Mutex<u64>,
    /// Whether the coordinator permits automatic execution-state collection.
    automatic_collection: AtomicBool,
    /// Evicted or uncached check/error type IDs awaiting deferred coordinator unloading.
    generated_types: Mutex<crate::coordinator::UnloadLibrary>,
}

impl JitController {
    fn cancellation_error(&self) -> tonic::Status {
        self.background_error
            .lock()
            .unwrap()
            .clone()
            .unwrap_or_else(|| tonic::Status::cancelled("JIT operation cancelled by runtime shutdown or reset"))
    }

    fn check_active(&self, token: &CancellationToken) -> Result<(), tonic::Status> {
        if token.is_cancelled() {
            Err(self.cancellation_error())
        } else {
            Ok(())
        }
    }

    fn record_background_error(&self, token: &CancellationToken, error: &tonic::Status, gid: u64, operation: &str) {
        let mut failure = self.background_error.lock().unwrap();
        // Reset/shutdown cancellation must not become a background failure.
        if failure.is_some() || token.is_cancelled() {
            return;
        }
        let error = tonic::Status::with_details_and_metadata(
            error.code(),
            format!("{operation} for gid={gid}: {}", error.message()),
            error.details().to_vec().into(),
            error.metadata().clone(),
        );
        eprintln!("{error}");
        *failure = Some(error);
        drop(failure);
        token.cancel();
    }

    async fn run_with_cancellation<T>(
        &self,
        token: &CancellationToken,
        work: impl Future<Output = Result<T, tonic::Status>>,
    ) -> Result<T, tonic::Status> {
        tokio::select! {
            biased;
            () = token.cancelled() => Err(self.cancellation_error()),
            result = work => {
                self.check_active(token)?;
                result
            }
        }
    }

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
            terminal_etypes: RwLock::new(HashMap::new()),
            library,
            undecoded_gids: RwLock::new(HashSet::new()),
            cancellation: RwLock::new(CancellationToken::new()),
            background_error: std::sync::Mutex::new(None),
            task_counter: TaskCounter::new(),
            collection: Arc::default(),
            collection_requests: AtomicU64::new(0),
            collection_completed: Mutex::new(0),
            automatic_collection: AtomicBool::new(false),
            generated_types: Mutex::default(),
        })
    }

    /// Create a JitController from an in-memory library.
    pub fn new_from_library(library: crate::jit::JitLibrary) -> Arc<Self> {
        let compiler = JitCompiler::new();
        Arc::new(Self {
            config: JitControllerConfig {
                filepath: None,
                cache_limit_ratio: default_cache_limit_ratio(),
            },
            compiler,
            coordinator: RwLock::new(None),
            type_cache: RwLock::new(TypeCache::new()),
            next_ctype: AtomicU64::new(1),
            next_etype: AtomicU64::new(1),
            terminal_etypes: RwLock::new(HashMap::new()),
            library,
            undecoded_gids: RwLock::new(HashSet::new()),
            cancellation: RwLock::new(CancellationToken::new()),
            background_error: std::sync::Mutex::new(None),
            task_counter: TaskCounter::new(),
            collection: Arc::default(),
            collection_requests: AtomicU64::new(0),
            collection_completed: Mutex::new(0),
            automatic_collection: AtomicBool::new(false),
            generated_types: Mutex::default(),
        })
    }
    pub async fn start(self: &Arc<Self>, client: CoordinatorClient) {
        let stats = client
            .memory_stats()
            .await
            .expect("failed to query coordinator collection policy");
        self.automatic_collection.store(stats.automatic_collection, Ordering::SeqCst);
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
        let token = self.cancellation.read().await;
        self.run_with_cancellation(&token, async {
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
            let error_model_types = self.register_terminal_types(&gadget_types).await;
            coordinator
                .load_library(bin::Library {
                    port_types,
                    gadget_types,
                    error_model_types,
                    ..Default::default()
                })
                .await
        })
        .await
    }

    async fn register_terminal_types(&self, gadget_types: &[bin::GadgetType]) -> Vec<bin::ErrorModelType> {
        let terminal_types = self.compiler.terminal_error_model_types.read().await;
        let mut terminal_etypes = self.terminal_etypes.write().await;
        let mut error_model_types = vec![];
        for gadget in gadget_types {
            let etype = self.next_etype();
            let mut terminal_type = terminal_types[&gadget.gtype].as_ref().clone();
            terminal_type.etype = etype;
            terminal_etypes.insert(gadget.gtype, etype);
            error_model_types.push(terminal_type);
        }
        error_model_types
    }

    pub fn next_ctype(&self) -> u64 {
        self.next_ctype.fetch_add(1, Ordering::SeqCst)
    }

    pub fn next_etype(&self) -> u64 {
        self.next_etype.fetch_add(1, Ordering::SeqCst)
    }

    /// Read cached types concurrently; recheck misses under the write lock before loading.
    async fn get_or_load_ctype(
        &self,
        check_model_type: &mut bin::CheckModelType,
        coordinator: &CoordinatorClient,
    ) -> Result<u64, tonic::Status> {
        if let Some(cached) = self.type_cache.read().await.get_check_model_type(check_model_type) {
            return Ok(cached);
        }
        let cache_limit = self.config.cache_limit(self.compiler.jit_gadget_types.read().await.len());
        let mut cache = self.type_cache.write().await;
        if let Some(cached) = cache.get_check_model_type(check_model_type) {
            return Ok(cached);
        }
        let ctype = self.next_ctype();
        check_model_type.ctype = ctype;
        coordinator
            .load_library(bin::Library {
                check_model_types: vec![check_model_type.clone()],
                ..Default::default()
            })
            .await?;
        if cache.check_model_types.len() >= cache_limit {
            let mut unused = self.generated_types.lock().await;
            while cache.check_model_types.len() >= cache_limit {
                let retired = cache.pop_check_model_type().expect("nonempty check-type cache");
                if self.automatic_collection.load(Ordering::SeqCst) {
                    unused.ctype.push(retired);
                }
            }
        }
        cache.insert_check_model_type(check_model_type, ctype);
        Ok(ctype)
    }

    /// Read cached types concurrently; recheck misses under the write lock before loading.
    async fn get_or_load_etype(
        &self,
        error_model_type: &Arc<bin::ErrorModelType>,
        coordinator: &CoordinatorClient,
    ) -> Result<u64, tonic::Status> {
        if let Some(cached) = self.type_cache.read().await.get_shared_error_model_type(error_model_type) {
            return Ok(cached);
        }
        let cache_limit = self.config.cache_limit(self.compiler.jit_gadget_types.read().await.len());
        let mut cache = self.type_cache.write().await;
        if let Some(cached) = cache.get_shared_error_model_type(error_model_type) {
            return Ok(cached);
        }
        if let Some(cached) = cache.get_error_model_type(error_model_type) {
            cache.remember_error_model_identity(error_model_type, cached, cache_limit);
            return Ok(cached);
        }
        let etype = self.next_etype();
        let mut declaration = error_model_type.as_ref().clone();
        declaration.etype = etype;
        coordinator
            .load_library(bin::Library {
                error_model_types: vec![declaration],
                ..Default::default()
            })
            .await?;
        if cache.error_model_types.len() >= cache_limit {
            let mut unused = self.generated_types.lock().await;
            while cache.error_model_types.len() >= cache_limit {
                let retired = cache.pop_error_model_type().expect("nonempty error-type cache");
                if self.automatic_collection.load(Ordering::SeqCst) {
                    unused.etype.push(retired);
                }
            }
        }
        cache.insert_error_model_type(error_model_type, etype);
        cache.remember_error_model_identity(error_model_type, etype, cache_limit);
        Ok(etype)
    }

    pub async fn clear_cache(&self) {
        let mut cache = self.type_cache.write().await;
        if self.automatic_collection.load(Ordering::SeqCst) {
            let mut unused = self.generated_types.lock().await;
            unused.ctype.extend(cache.check_model_types.values().copied());
            unused.etype.extend(cache.error_model_types.values().copied());
        }
        cache.clear();
    }

    /// Execute a JIT instruction, compiling it and sending commands to the coordinator.
    ///
    /// Returns the assigned gid for the instantiated gadget once the gadget and check model
    /// are loaded. The error model is loaded asynchronously in the background to avoid
    /// circular dependencies (error models depend on future gadgets' gids).
    /// # Panics
    /// Panics if execution fails. Use [`Self::try_execute`] to receive errors without panicking.
    pub async fn execute(self: &Arc<Self>, instruction: jit::JitInstruction) -> u64 {
        self.try_execute(instruction).await.expect("JIT execution failed")
    }

    /// Compile an instruction after checking that its references are still live.
    ///
    /// # Errors
    /// Returns validation, execution, or the first background error in this shot.
    pub async fn try_execute(self: &Arc<Self>, instruction: jit::JitInstruction) -> Result<u64, tonic::Status> {
        let token = self.cancellation.read().await;
        self.run_with_cancellation(&token, async {
            let _collection = self.collection.read().await;
            let gadget = instruction
                .gadget
                .as_ref()
                .ok_or_else(|| tonic::Status::invalid_argument("missing gadget"))?;
            if gadget.gid != 0
                && (self.compiler.was_collected(gadget.gid).await || self.compiler.contains_gid(gadget.gid).await)
            {
                return Err(tonic::Status::invalid_argument(
                    "gadget id already exists or has been collected",
                ));
            }
            for connector in &gadget.connectors {
                if !self.compiler.contains_gid(connector.gid).await {
                    return Err(tonic::Status::invalid_argument(
                        "connector refers to an unknown or retired gadget",
                    ));
                }
            }
            Box::pin(self.execute_with_token(instruction, token.clone())).await
        })
        .await
    }

    async fn collect_garbage(&self) -> Result<(), tonic::Status> {
        if !self.automatic_collection.load(Ordering::SeqCst) {
            return Ok(());
        }
        self.collection_requests.fetch_add(1, Ordering::AcqRel);
        let mut completed = self.collection_completed.lock().await;
        if *completed == self.collection_requests.load(Ordering::Acquire) {
            return Ok(());
        }
        let gadgets = self.compiler.gadgets.write().await;
        // Include mutations completed while the graph snapshot was queued.
        let requested = self.collection_requests.load(Ordering::Acquire);
        self.compiler.collect_garbage(gadgets).await;
        self.collect_unused_generated_types().await?;
        *completed = requested;
        Ok(())
    }

    async fn collect_unused_generated_types(&self) -> Result<(), tonic::Status> {
        if !self.automatic_collection.load(Ordering::SeqCst) {
            return Ok(());
        }
        {
            let unused = self.generated_types.lock().await;
            if unused.gtype.is_empty() && unused.ctype.is_empty() && unused.etype.is_empty() {
                return Ok(());
            }
        }
        // Graph publication is atomic; only type removal needs this lifetime barrier.
        let _collection = self.collection.write().await;
        let coordinator = self.coordinator.read().await;
        let coordinator = coordinator
            .as_ref()
            .ok_or_else(|| tonic::Status::failed_precondition("coordinator not connected"))?;
        let mut unload = self.generated_types.lock().await;
        if unload.gtype.is_empty() && unload.ctype.is_empty() && unload.etype.is_empty() {
            return Ok(());
        }
        // Keep the IDs pending if the request fails or its caller is cancelled.
        coordinator.unload(unload.clone()).await?;
        *unload = crate::coordinator::UnloadLibrary::default();
        Ok(())
    }

    /// Inspect coordinator and compiler state without changing object lifetimes.
    ///
    /// # Errors
    /// Returns a status if the coordinator is disconnected or unavailable.
    pub async fn memory_stats(&self) -> Result<crate::coordinator::MemoryStats, tonic::Status> {
        let coordinator = self.coordinator.read().await;
        let coordinator = coordinator
            .as_ref()
            .ok_or_else(|| tonic::Status::failed_precondition("coordinator not connected"))?;
        let mut stats = coordinator.memory_stats().await?;
        stats.jit_gadgets = self.compiler.gadgets.read().await.len() as u64;
        Ok(stats)
    }

    async fn execute_with_token(
        self: &Arc<Self>,
        mut instruction: jit::JitInstruction,
        token: CancellationToken,
    ) -> Result<u64, tonic::Status> {
        let input = instruction
            .gadget
            .as_mut()
            .ok_or_else(|| tonic::Status::invalid_argument("missing gadget"))?;
        self.compiler.assign_gid(input);
        {
            let coordinator = self.coordinator.read().await;
            let coordinator = coordinator
                .as_ref()
                .ok_or_else(|| tonic::Status::failed_precondition("coordinator not connected"))?;
            let response = coordinator
                .execute(bin::Instruction {
                    create: Some(bin::instruction::Create::Gadget(input.clone())),
                })
                .await?;
            if response.id != input.gid {
                return Err(tonic::Status::internal("coordinator changed the assigned gadget id"));
            }
        }
        let (gadget, mut check_model_type, mut check_model, error_model_future) =
            Arc::clone(&self.compiler).compile_shared(instruction, token.clone()).await;

        let gid = gadget.gid;
        if let Some(terminal) = check_model.terminal_error_model.as_mut() {
            terminal.etype = self.terminal_etypes.read().await[&gadget.gtype];
        }
        let cid = gid;

        self.undecoded_gids.write().await.insert(gid);

        {
            let coordinator_guard = self.coordinator.read().await;
            let coordinator = coordinator_guard.as_ref().expect("coordinator not connected");

            let ctype = self.get_or_load_ctype(&mut check_model_type, coordinator).await?;

            let check_model_modifier = build_check_model_modifier(&check_model_type);
            check_model.ctype = ctype;
            check_model.modifier = check_model_modifier;
            coordinator
                .execute(bin::Instruction {
                    create: Some(bin::instruction::Create::CheckModel(check_model)),
                })
                .await?;
        }

        // Spawn error model loading in background. The JIT compiler determines when the
        // error model future resolves (once output ports are connected), and we send it
        // to the coordinator as soon as it's ready.
        let this = Arc::clone(self);
        let _guard = self.task_counter.guard();
        tokio::spawn(async move {
            let _guard = _guard;
            let attachment = async {
                let (error_model_type, error_model) = error_model_future.await;
                let collection = this.collection.read().await;

                let coordinator_guard = this.coordinator.read().await;
                let coordinator = coordinator_guard
                    .as_ref()
                    .ok_or_else(|| tonic::Status::failed_precondition("coordinator not connected"))?;

                let etype = this.get_or_load_etype(&error_model_type, coordinator).await?;

                let error_model_modifier = build_error_model_modifier(&error_model_type, &error_model);
                drop(error_model_type);
                coordinator
                    .execute(bin::Instruction {
                        create: Some(bin::instruction::Create::ErrorModel(bin::ErrorModel {
                            etype,
                            cid,
                            tag: error_model.tag.clone(),
                            modifier: error_model_modifier,
                            eid: error_model.eid,
                        })),
                    })
                    .await?;
                this.compiler.mark_error_model_ready(gid).await;
                drop(coordinator_guard);
                drop(collection);
                Ok(())
            };
            if let Err(error) = this.run_with_cancellation(&token, attachment).await {
                this.record_background_error(&token, &error, gid, "failed to attach error model");
                return;
            }
            if let Err(error) = this.collect_garbage().await {
                this.record_background_error(&token, &error, gid, "failed to collect JIT state");
            }
        });

        Ok(gid)
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
    /// Returns `BatchExecuteError` for validation, execution, cancellation, or a background failure.
    ///
    /// # Panics
    ///
    /// Panics if any spawned task panics during execution.
    pub async fn batch_execute(
        self: &Arc<Self>,
        instructions: Vec<jit::JitInstruction>,
    ) -> Result<Vec<u64>, BatchExecuteError> {
        let token = self.cancellation.read().await;
        self.check_active(&token).map_err(BatchExecuteError::Execution)?;
        let collection = self
            .run_with_cancellation(&token, async {
                Ok(Arc::new(Arc::clone(&self.collection).read_owned().await))
            })
            .await
            .map_err(BatchExecuteError::Execution)?;
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
            if self.compiler.was_collected(gid).await {
                return Err(BatchExecuteError::RetiredGid { gid });
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
            for dep_gid in gadget.submission_dependencies() {
                if let Some(&dep_index) = gid_to_index.get(&dep_gid) {
                    dependencies[index].push(dep_index);
                }
            }
        }
        let mut remaining: Vec<_> = dependencies.iter().map(Vec::len).collect();
        let mut successors = vec![vec![]; instructions.len()];
        for (index, dependencies) in dependencies.iter().enumerate() {
            for &dependency in dependencies {
                successors[dependency].push(index);
            }
        }
        let mut ready: Vec<_> = remaining
            .iter()
            .enumerate()
            .filter_map(|(index, &count)| (count == 0).then_some(index))
            .collect();
        let mut visited = 0;
        while let Some(index) = ready.pop() {
            visited += 1;
            for &successor in &successors[index] {
                remaining[successor] -= 1;
                if remaining[successor] == 0 {
                    ready.push(successor);
                }
            }
        }
        if visited != instructions.len() {
            return Err(BatchExecuteError::CyclicDependencies);
        }

        let gids: Vec<u64> = instructions.iter().map(|i| i.gadget.as_ref().unwrap().gid).collect();
        let num_instructions = instructions.len();
        let instructions = Arc::new(instructions);
        // Use watch channels so that late subscribers still see the completion
        // signal (broadcast channels drop messages if no subscriber exists yet).
        let completion_txs: Arc<Vec<_>> =
            Arc::new((0..num_instructions).map(|_| tokio::sync::watch::channel(false).0).collect());

        let mut handles = Vec::with_capacity(num_instructions);
        for index in 0..num_instructions {
            let this = Arc::clone(self);
            let instructions = Arc::clone(&instructions);
            let deps = dependencies[index].clone();
            let completion_txs = Arc::clone(&completion_txs);
            let collection = Arc::clone(&collection);
            let token = token.clone();
            let task_guard = self.task_counter.guard();

            let handle = tokio::spawn(async move {
                let _task_guard = task_guard;
                let _collection = collection;
                let execution = async {
                    for &dep_index in &deps {
                        let mut rx = completion_txs[dep_index].subscribe();
                        rx.wait_for(|&done| done)
                            .await
                            .map_err(|_| tonic::Status::internal("JIT dependency completion channel closed"))?;
                    }

                    let instruction = instructions[index].clone();
                    let gid = Box::pin(this.execute_with_token(instruction, token.clone())).await?;
                    completion_txs[index].send_replace(true);
                    Ok(gid)
                };
                let result = this.run_with_cancellation(&token, execution).await;
                if let Err(error) = &result {
                    this.record_background_error(
                        &token,
                        error,
                        instructions[index].gadget.as_ref().unwrap().gid,
                        "failed to execute JIT batch instruction",
                    );
                }
                result
            });
            handles.push(handle);
        }

        let completed = futures_util::future::join_all(handles).await;
        self.check_active(&token).map_err(BatchExecuteError::Execution)?;
        let results: Vec<u64> = completed
            .into_iter()
            .map(|r| r.expect("batch execute task panicked").map_err(BatchExecuteError::Execution))
            .collect::<Result<_, _>>()?;

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
        let token = self.cancellation.read().await;
        self.check_active(&token)?;
        let handles: Vec<_> = outcomes
            .into_iter()
            .map(|outcome| {
                let this = Arc::clone(&this);
                let token = token.clone();
                let task_guard = self.task_counter.guard();
                tokio::spawn(async move {
                    let _task_guard = task_guard;
                    this.decode_single_with_token(outcome, token).await
                })
            })
            .collect();

        let completed = futures_util::future::join_all(handles).await;
        self.check_active(&token)?;
        completed
            .into_iter()
            .map(|result| result.expect("batch decode task panicked"))
            .collect()
    }

    pub async fn decode_single(
        self: &Arc<Self>,
        outcomes: crate::coordinator::Outcomes,
    ) -> Result<crate::coordinator::Readouts, tonic::Status> {
        let token = self.cancellation.read().await;
        self.decode_single_with_token(outcomes, token.clone()).await
    }

    async fn decode_single_with_token(
        self: &Arc<Self>,
        outcomes: crate::coordinator::Outcomes,
        token: CancellationToken,
    ) -> Result<crate::coordinator::Readouts, tonic::Status> {
        self.run_with_cancellation(&token, async {
            let gid = outcomes.gid;

            if !self.undecoded_gids.write().await.remove(&gid) {
                return Err(tonic::Status::invalid_argument(format!(
                    "decode called for unknown or already-decoded gid: {gid}"
                )));
            }
            let coordinator_guard = self.coordinator.read().await;
            let coordinator = coordinator_guard
                .as_ref()
                .ok_or_else(|| tonic::Status::failed_precondition("coordinator not connected"))?;
            let result = coordinator.decode(outcomes).await;
            drop(coordinator_guard);
            if result.is_ok() {
                self.collect_garbage().await?;
            }
            result
        })
        .await
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
        let mut token = self.cancellation.write().await;
        self.task_counter.wait_for_zero().await;
        let reset_library = flags.reset_library;
        if reset_library {
            self.compiler.reset_library().await;
            self.terminal_etypes.write().await.clear();
            self.compiler.load_library(self.library.clone()).await;
            self.clear_cache().await;
        } else {
            self.compiler.reset().await;
        }
        self.undecoded_gids.write().await.clear();
        let unused_types = self.generated_types.lock().await.clone();
        let coordinator_guard = self.coordinator.read().await;
        if let Some(coordinator) = coordinator_guard.as_ref() {
            coordinator.reset(flags).await?;
            if self.automatic_collection.load(Ordering::SeqCst) {
                coordinator.unload(unused_types).await?;
            }

            // Re-send port types and gadget types so the coordinator stays consistent
            // with what the JIT controller expects to reference in future compiles.
            if reset_library {
                self.next_ctype.store(1, Ordering::SeqCst);
                self.next_etype.store(1, Ordering::SeqCst);
                let port_types: Vec<_> = self.library.port_types.iter().map(|pt| pt.base.clone().unwrap()).collect();
                let gadget_types: Vec<_> = self.library.gadget_types.iter().map(|gt| gt.base.clone().unwrap()).collect();
                let error_model_types = self.register_terminal_types(&gadget_types).await;
                coordinator
                    .load_library(bin::Library {
                        port_types,
                        gadget_types,
                        error_model_types,
                        ..Default::default()
                    })
                    .await?;
            }
        }
        *self.generated_types.lock().await = crate::coordinator::UnloadLibrary::default();
        *self.background_error.lock().unwrap() = None;
        *token = CancellationToken::new();
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
    CyclicDependencies,
    Execution(tonic::Status),
    RetiredGid { gid: u64 },
    MissingGadget { index: usize },
    ZeroGid { index: usize },
    DuplicateGid { gid: u64, index: usize },
    MissingDependency { index: usize, gid: u64, missing_gid: u64 },
}

impl std::fmt::Display for BatchExecuteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CyclicDependencies => write!(f, "cyclic gadget dependencies"),
            Self::Execution(error) => write!(f, "JIT execution failed: {error}"),
            Self::RetiredGid { gid } => write!(f, "gid={gid} belongs to retired history"),
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
        match err {
            BatchExecuteError::Execution(status) => status,
            validation => tonic::Status::invalid_argument(validation.to_string()),
        }
    }
}

/// Wrapper that holds Arc<JitController> for implementing the gRPC trait.
/// This allows gRPC methods to call Arc-based methods like `execute`.
#[cfg(feature = "cli")]
struct JitControllerService(Arc<JitController>);

#[cfg(feature = "cli")]
#[tonic::async_trait]
impl jit_controller_server::JitController for JitControllerService {
    async fn get_memory_stats(
        &self,
        _request: tonic::Request<()>,
    ) -> Result<tonic::Response<crate::coordinator::MemoryStats>, tonic::Status> {
        self.0.memory_stats().await.map(tonic::Response::new)
    }

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
        let gid = self.0.try_execute(instruction).await?;
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
        let readouts = self.0.decode_single(request.into_inner()).await?;
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
    let mut reroutes: Vec<_> = error_model_type
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
    if let Some(modifier) = &error_model.modifier {
        for reroute in &modifier.reroute_remote_check_models {
            let index = usize::try_from(reroute.remote_check_model_index).unwrap();
            *reroutes
                .get_mut(index)
                .expect("compiled error-model binding must be in range") = reroute.clone();
        }
    }

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
        check_model_types_eq(&self.0, &other.0)
    }
}

impl Eq for CheckModelTypeKey {}

impl Hash for CheckModelTypeKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        hash_check_model_type_structural(&self.0, state);
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
        error_model_types_eq(&self.0, &other.0)
    }
}

impl Eq for ErrorModelTypeKey {}

impl Hash for ErrorModelTypeKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        hash_error_model_type_structural(&self.0, state);
    }
}

fn check_model_types_eq(a: &bin::CheckModelType, b: &bin::CheckModelType) -> bool {
    a.gtype == b.gtype
        && a.checks.len() == b.checks.len()
        && a.checks.iter().zip(&b.checks).all(|(left, right)| checks_eq(left, right))
}

fn error_model_types_eq(a: &bin::ErrorModelType, b: &bin::ErrorModelType) -> bool {
    a.ctype == b.ctype
        && a.errors.len() == b.errors.len()
        && a.errors.iter().zip(&b.errors).all(|(left, right)| errors_eq(left, right))
}

fn hash_check_model_type_structural<H: Hasher>(model: &bin::CheckModelType, state: &mut H) {
    model.gtype.hash(state);
    model.checks.len().hash(state);
    for check in &model.checks {
        hash_check(check, state);
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
pub struct TypeCache {
    check_model_types: HashMap<Arc<CheckModelTypeKey>, u64>,
    error_model_types: HashMap<Arc<ErrorModelTypeKey>, u64>,
    // Structural indices avoid cloning lookup keys; ID-keyed LRUs also promote identity hits.
    check_model_recency: std::sync::Mutex<lru::LruCache<u64, Arc<CheckModelTypeKey>>>,
    error_model_recency: std::sync::Mutex<lru::LruCache<u64, Arc<ErrorModelTypeKey>>>,
    /// Allocation-address shortcuts, bounded by the same budget as the structural error cache.
    error_model_identities: std::sync::Mutex<lru::LruCache<usize, (Weak<bin::ErrorModelType>, u64)>>,
}

impl Default for TypeCache {
    fn default() -> Self {
        Self {
            check_model_types: HashMap::new(),
            error_model_types: HashMap::new(),
            // The controller applies its library-dependent budget using pop_lru,
            // retaining every evicted ID for deferred coordinator unloading.
            check_model_recency: std::sync::Mutex::new(lru::LruCache::unbounded()),
            error_model_recency: std::sync::Mutex::new(lru::LruCache::unbounded()),
            error_model_identities: std::sync::Mutex::new(lru::LruCache::new(std::num::NonZeroUsize::MIN)),
        }
    }
}

impl TypeCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Look up a structurally equivalent check type and refresh its recency.
    ///
    /// # Panics
    /// Panics if the check-type recency mutex is poisoned.
    pub fn get_check_model_type(&self, check_model_type: &bin::CheckModelType) -> Option<u64> {
        let mut hasher = self.check_model_types.hasher().build_hasher();
        hash_check_model_type_structural(check_model_type, &mut hasher);
        let id = self
            .check_model_types
            .raw_entry()
            .from_hash(hasher.finish(), |key| check_model_types_eq(&key.0, check_model_type))
            .map(|(_, &id)| id)?;
        self.check_model_recency.lock().unwrap().promote(&id);
        Some(id)
    }

    /// Register a check type as the most recently used entry.
    ///
    /// # Panics
    /// Panics if the check-type recency mutex is poisoned.
    pub fn insert_check_model_type(&mut self, check_model_type: &bin::CheckModelType, ctype: u64) {
        let key = Arc::new(CheckModelTypeKey(check_model_type.clone()));
        let recency = self.check_model_recency.get_mut().unwrap();
        if let Some(previous_key) = recency.pop(&ctype) {
            self.check_model_types.remove(&previous_key);
        }
        if let Some(previous) = self.check_model_types.insert(Arc::clone(&key), ctype) {
            recency.pop(&previous);
        }
        recency.put(ctype, key);
    }

    /// Look up a structurally equivalent error type and refresh its recency.
    ///
    /// # Panics
    /// Panics if the error-type recency mutex is poisoned.
    pub fn get_error_model_type(&self, error_model_type: &bin::ErrorModelType) -> Option<u64> {
        let mut hasher = self.error_model_types.hasher().build_hasher();
        hash_error_model_type_structural(error_model_type, &mut hasher);
        let id = self
            .error_model_types
            .raw_entry()
            .from_hash(hasher.finish(), |key| error_model_types_eq(&key.0, error_model_type))
            .map(|(_, &id)| id)?;
        self.error_model_recency.lock().unwrap().promote(&id);
        Some(id)
    }

    /// Register an error type as the most recently used entry.
    ///
    /// # Panics
    /// Panics if the error-type recency or identity-cache mutex is poisoned.
    pub fn insert_error_model_type(&mut self, error_model_type: &bin::ErrorModelType, etype: u64) {
        let key = Arc::new(ErrorModelTypeKey(error_model_type.clone()));
        if let Some(previous_key) = self.error_model_recency.get_mut().unwrap().pop(&etype) {
            self.error_model_types.remove(&previous_key);
            self.forget_error_model_identities(etype);
        }
        if let Some(previous) = self.error_model_types.insert(Arc::clone(&key), etype) {
            self.error_model_recency.get_mut().unwrap().pop(&previous);
            self.forget_error_model_identities(previous);
        }
        self.error_model_recency.get_mut().unwrap().put(etype, key);
    }

    fn pop_check_model_type(&mut self) -> Option<u64> {
        let (id, key) = self.check_model_recency.get_mut().unwrap().pop_lru()?;
        self.check_model_types.remove(&key);
        Some(id)
    }

    fn pop_error_model_type(&mut self) -> Option<u64> {
        let (id, key) = self.error_model_recency.get_mut().unwrap().pop_lru()?;
        self.error_model_types.remove(&key);
        self.forget_error_model_identities(id);
        Some(id)
    }

    fn get_shared_error_model_type(&self, model: &Arc<bin::ErrorModelType>) -> Option<u64> {
        let id = self
            .error_model_identities
            .lock()
            .unwrap()
            .get(&Arc::as_ptr(model).addr())
            .map(|(_, id)| *id)?;
        self.error_model_recency.lock().unwrap().promote(&id).then_some(id)
    }

    fn remember_error_model_identity(&mut self, model: &Arc<bin::ErrorModelType>, etype: u64, cache_limit: usize) {
        let identities = self.error_model_identities.get_mut().unwrap();
        identities.resize(std::num::NonZeroUsize::new(cache_limit).expect("positive identity cache budget"));
        // The weak reference prevents address reuse, and Arc mutation detaches from it.
        identities.put(Arc::as_ptr(model).addr(), (Arc::downgrade(model), etype));
    }

    fn forget_error_model_identities(&mut self, etype: u64) {
        let identities = self.error_model_identities.get_mut().unwrap();
        let retired: Vec<_> = identities
            .iter()
            .filter_map(|(&address, (_, id))| (*id == etype).then_some(address))
            .collect();
        for address in retired {
            identities.pop(&address);
        }
    }

    /// Remove all structural entries, recency records, and identity aliases.
    ///
    /// # Panics
    /// Panics if a recency or identity-cache mutex is poisoned.
    pub fn clear(&mut self) {
        self.check_model_types.clear();
        self.error_model_types.clear();
        self.check_model_recency.get_mut().unwrap().clear();
        self.error_model_recency.get_mut().unwrap().clear();
        self.error_model_identities.get_mut().unwrap().clear();
    }
}

#[cfg(test)]
#[path = "../../tests/unit/jit_controller_cache_test.rs"]
mod cache_tests;
