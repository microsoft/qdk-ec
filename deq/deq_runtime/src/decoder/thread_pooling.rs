//! Decoder with thread pooling
//!

use crate::decoder::blackbox_decoder::{self, ParityFactor, black_box_decoder_server};
pub use crate::decoder::decoder_features::DecoderFeatures;
use crate::misc::bit_vector;
#[cfg(feature = "cli")]
use crate::misc::fastrace::{Event, Span, SpanContext};
use crate::misc::validation;
use crate::util::BitVector;
use blackbox_decoder::DecodingHypergraph;
use hashbrown::HashMap;
use serde::{Deserialize, Serialize};
use std::collections::LinkedList;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
#[cfg(feature = "cli")]
use structdoc::StructDoc;
use tokio::sync::{Mutex, OwnedSemaphorePermit, Semaphore, oneshot, watch};
#[cfg(feature = "cli")]
use tonic::transport::server::Router;
use tonic::{Request, Response, Status};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "cli", derive(StructDoc))]
pub struct ThreadPoolingConfig {
    /// number of parallel threads in the pool, default to number of logical cores
    #[serde(default)]
    pub parallel: usize,
}

pub struct ThreadPoolingDecoder<T: DecoderInstance> {
    pub config: ThreadPoolingConfig,
    pub original_config: Arc<serde_json::Value>,
    pub thread_pool: Arc<rayon::ThreadPool>,
    work_slots: Arc<Semaphore>,
    caller: Option<std::thread::Thread>,
    loaded: Arc<Mutex<HashMap<u64, Loaded<T>>>>,
    next_hid: AtomicU64,
    decoding: watch::Sender<usize>,
    features: DecoderFeatures,
    prepared_hypergraphs: std::sync::Mutex<lru::LruCache<usize, Arc<PreparedHypergraph<T>>>>,
}

/// Immutable backend setup that constructs a fresh, independent decoder on each call.
/// Factories receive the same graph and configuration that were prepared.
pub type DecoderFactory<T> = Arc<dyn Fn(&DecodingHypergraph, &serde_json::Value) -> T + Send + Sync>;

struct PreparedHypergraph<T> {
    hypergraph: std::sync::Weak<DecodingHypergraph>,
    configuration: std::sync::Weak<serde_json::Value>,
    factory: OnceLock<Result<DecoderFactory<T>, String>>,
}

impl<T: DecoderInstance + 'static> PreparedHypergraph<T> {
    fn matches(&self, hypergraph: &Arc<DecodingHypergraph>, configuration: &Arc<serde_json::Value>) -> bool {
        std::ptr::eq(self.hypergraph.as_ptr(), Arc::as_ptr(hypergraph))
            && std::ptr::eq(self.configuration.as_ptr(), Arc::as_ptr(configuration))
    }

    fn new_instance(
        &self,
        hypergraph: &Arc<DecodingHypergraph>,
        configuration: &Arc<serde_json::Value>,
    ) -> Result<T, DecodeError> {
        if !self.matches(hypergraph, configuration) {
            return Err(DecodeError::InvalidInput(
                "prepared decoder graph or configuration changed".into(),
            ));
        }
        self.factory
            .get_or_init(|| T::prepare(hypergraph, configuration))
            .as_ref()
            .map(|factory| factory(hypergraph, configuration))
            .map_err(|error| DecodeError::InvalidInput(error.clone()))
    }
}

pub struct Loaded<T: DecoderInstance> {
    hypergraph: Arc<DecodingHypergraph>,
    prepared: Arc<PreparedHypergraph<T>>,
    instances: LinkedList<T>,
}

// Queued requests count toward reset; admitted work retains its slot through worker completion.
struct DecodingGuard {
    tx: watch::Sender<usize>,
    slot: Option<OwnedSemaphorePermit>,
}

impl DecodingGuard {
    fn new(tx: watch::Sender<usize>) -> Self {
        Self { tx, slot: None }
    }
}

impl Drop for DecodingGuard {
    fn drop(&mut self) {
        drop(self.slot.take());
        self.tx.send_modify(|v| *v -= 1);
    }
}

#[cfg(feature = "cli")]
fn finish_worker_span(span: Span, offsets: [u128; 4], success: Option<bool>) {
    if let Some(success) = success {
        drop(span.with_properties(|| {
            [
                "worker_started_offset_ns",
                "initialized_offset_ns",
                "solve_finished_offset_ns",
                "worker_finished_offset_ns",
            ]
            .into_iter()
            .zip(offsets)
            .map(|(name, offset)| (name, offset.to_string()))
            .chain(std::iter::once(("solver_success", success.to_string())))
        }));
    }
}

impl<T: DecoderInstance> std::fmt::Debug for Loaded<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Loaded")
            .field("hypergraph", &self.hypergraph)
            .finish_non_exhaustive()
    }
}

pub struct DecodeRequest<'a> {
    pub syndrome: &'a BitVector,
    /// Optional deterministic seed for decoder randomness. Zero is valid.
    pub decoder_seed: Option<u64>,
    /// Shot-scoped prior assignments. Implementations must not let these values
    /// affect a later request served by the same pooled instance.
    pub reweights: &'a [(u64, f64)],
    /// Structured loss observation for this shot. Independent of `reweights`;
    /// a decoder advertising both features must accept both together.
    pub loss: Option<&'a blackbox_decoder::LossInfo>,
}

impl DecodeRequest<'_> {
    fn required_features(&self) -> DecoderFeatures {
        DecoderFeatures::required(self.decoder_seed.is_some(), !self.reweights.is_empty(), self.loss.is_some())
    }

    fn require_supported(&self, supported: DecoderFeatures) -> Result<(), DecodeError> {
        self.required_features()
            .require_supported_by(supported)
            .map_err(DecodeError::Unsupported)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecodeError {
    Unsupported(DecoderFeatures),
    InvalidInput(String),
    Backend(String),
}

impl fmt::Display for DecodeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsupported(features) => write!(formatter, "unsupported decoder features: {features}"),
            Self::InvalidInput(message) => write!(formatter, "invalid decode request: {message}"),
            Self::Backend(message) => write!(formatter, "decoder backend failed: {message}"),
        }
    }
}

impl std::error::Error for DecodeError {}

pub trait DecoderInstance {
    #[must_use]
    fn supported_features(_config: &serde_json::Value) -> DecoderFeatures {
        DecoderFeatures::empty()
    }

    /// Validates constraints imposed by a specific decoder backend.
    ///
    /// # Errors
    ///
    /// Returns an error when the hypergraph cannot be represented by the backend.
    fn validate_hypergraph(_hypergraph: &DecodingHypergraph, _config: &serde_json::Value) -> Result<(), String> {
        Ok(())
    }

    fn new(hypergraph: &DecodingHypergraph, config: &serde_json::Value) -> Self;

    /// Prepare reusable immutable setup for a validated graph and configuration.
    /// The factory must create independent mutable state for every instance.
    /// Backends without separate setup reuse their ordinary constructor.
    ///
    /// # Errors
    /// Returns an error when backend configuration or initialization data is invalid.
    fn prepare(_hypergraph: &DecodingHypergraph, _config: &serde_json::Value) -> Result<DecoderFactory<Self>, String>
    where
        Self: Sized + 'static,
    {
        Ok(Arc::new(Self::new))
    }

    fn decode(&mut self, request: DecodeRequest<'_>) -> Result<ParityFactor, DecodeError>;

    fn reset(&mut self);
}

impl<T: DecoderInstance + Send + 'static> ThreadPoolingDecoder<T> {
    async fn acquire_work_slot(&self) -> Result<DecodingGuard, Status> {
        self.decoding.send_modify(|count| *count += 1);
        let mut guard = DecodingGuard::new(self.decoding.clone());
        guard.slot = Some(
            Arc::clone(&self.work_slots)
                .acquire_owned()
                .await
                .map_err(|_| Status::unavailable("decoder execution closed"))?,
        );
        Ok(guard)
    }

    pub(crate) async fn unload_hypergraphs(&self, hids: &[u64]) {
        let mut loaded = self.loaded.lock().await;
        let mut prepared = self.prepared_hypergraphs.lock().unwrap();
        for hid in hids {
            if let Some(graph) = loaded.remove(hid) {
                prepared.pop(&Arc::as_ptr(&graph.hypergraph).addr());
            }
        }
    }

    fn schedule_work(&self, work: impl FnOnce() + Send + 'static) {
        if self.thread_pool.current_thread_index().is_some() && crate::misc::pool_executor::inline_cpu_allowed() {
            crate::misc::pool_executor::flush_local(&self.thread_pool);
            work();
        } else {
            self.thread_pool.spawn(move || crate::misc::pool_executor::execute_cpu(work));
            if let Some(caller) = &self.caller {
                caller.unpark();
            }
        }
    }

    #[cfg(all(test, feature = "cli"))]
    pub(crate) fn pending_decodes(&self) -> watch::Receiver<usize> {
        self.decoding.subscribe()
    }

    #[must_use]
    pub fn features(&self) -> DecoderFeatures {
        self.features
    }

    /// # Panics
    ///
    /// Panics when `original_config` is invalid for thread pooling or the Rayon
    /// pool cannot be created.
    #[must_use]
    pub fn new(original_config: serde_json::Value) -> Self {
        Self::with_thread_pool(original_config, None)
    }

    /// Construct a decoder using an existing worker pool, or create its own.
    ///
    /// A caller-participating pool can eliminate ingress handoffs without
    /// increasing the configured number of computation threads.
    /// Construct on that caller thread so queued jobs can wake its parked driver.
    ///
    /// # Panics
    /// Panics on invalid configuration or failure to create a worker pool.
    #[must_use]
    pub fn with_thread_pool(original_config: serde_json::Value, thread_pool: Option<Arc<rayon::ThreadPool>>) -> Self {
        let mut config: ThreadPoolingConfig = serde_json::from_value(original_config.clone()).unwrap();
        let features = T::supported_features(&original_config);
        let thread_pool = if let Some(thread_pool) = thread_pool {
            config.parallel = thread_pool.current_num_threads();
            thread_pool
        } else {
            let mut thread_pool_builder = rayon::ThreadPoolBuilder::new();
            if config.parallel != 0 {
                thread_pool_builder = thread_pool_builder.num_threads(config.parallel);
            }
            Arc::new(
                thread_pool_builder
                    .panic_handler(|error| {
                        eprintln!("rayon pool thread panicked: {error:?}");
                    })
                    .build()
                    .expect("creating thread pool failed"),
            )
        };
        let caller = thread_pool.current_thread_index().map(|_| std::thread::current());
        let work_slots = Arc::new(Semaphore::new(thread_pool.current_num_threads()));
        Self {
            config,
            original_config: Arc::new(original_config),
            thread_pool,
            work_slots,
            caller,
            loaded: Default::default(),
            next_hid: AtomicU64::new(1),
            decoding: watch::channel(0).0,
            features,
        }
    }

    #[cfg(feature = "cli")]
    #[must_use]
    pub fn add_service(self: &Arc<Self>, router: Router) -> Router {
        let service =
            black_box_decoder_server::BlackBoxDecoderServer::from_arc(self.clone()).max_decoding_message_size(usize::MAX);
        router.add_service(service)
    }
}

impl<T: DecoderInstance + Send + 'static> ThreadPoolingDecoder<T> {
    fn prepare_shared_hypergraph(&self, hypergraph: &Arc<DecodingHypergraph>) -> Result<Arc<PreparedHypergraph<T>>, Status> {
        let identity = Arc::as_ptr(hypergraph).addr();
        if let Some(prepared) = self
            .prepared_hypergraphs
            .lock()
            .unwrap()
            .get(&identity)
            .filter(|cached| cached.matches(hypergraph, &self.original_config))
            .cloned()
        {
            return Ok(prepared);
        }
        validation::validate_hypergraph(hypergraph).map_err(Status::invalid_argument)?;
        T::validate_hypergraph(hypergraph, &self.original_config).map_err(Status::invalid_argument)?;
        let mut cache = self.prepared_hypergraphs.lock().unwrap();
        if let Some(prepared) = cache
            .get(&identity)
            .filter(|cached| cached.matches(hypergraph, &self.original_config))
        {
            return Ok(Arc::clone(prepared));
        }
        // Weak ownership prevents address reuse and makes Arc mutation detach.
        let prepared = Arc::new(PreparedHypergraph {
            hypergraph: Arc::downgrade(hypergraph),
            configuration: Arc::downgrade(&self.original_config),
            factory: OnceLock::new(),
        });
        // An exclusively owned one-shot graph cannot hit the identity cache again.
        if Arc::strong_count(hypergraph) > 1 {
            cache.put(identity, Arc::clone(&prepared));
        }
        Ok(prepared)
    }

    pub(crate) async fn decode_shared(
        &self,
        hypergraph: Arc<DecodingHypergraph>,
        syndrome: BitVector,
        loss: Option<blackbox_decoder::LossInfo>,
        decoder_seed: Option<u64>,
    ) -> Result<ParityFactor, Status> {
        let prepared = self.prepare_shared_hypergraph(&hypergraph)?;
        validation::validate_syndrome(&syndrome, hypergraph.vertex_num).map_err(Status::invalid_argument)?;
        validation::validate_loss(loss.as_ref(), hypergraph.hyperedges.len()).map_err(Status::invalid_argument)?;
        let request = DecodeRequest {
            syndrome: &syndrome,
            decoder_seed,
            reweights: &[],
            loss: loss.as_ref(),
        };
        request.require_supported(self.features).map_err(decode_error_status)?;
        // Observable metadata and optional request fields can affect a
        // zero-syndrome correction.
        if bit_vector::is_zero(&syndrome)
            && request.required_features().is_empty()
            && !self.features.contains(DecoderFeatures::OBSERVABLES)
        {
            #[cfg(feature = "cli")]
            let _span = Span::enter_with_local_parent("decoder_zero_syndrome");
            return Ok(ParityFactor { subgraph: vec![] });
        }
        let (tx, rx) = oneshot::channel::<Result<ParityFactor, DecodeError>>();
        let original_config = self.original_config.clone();
        self.decoding.send_modify(|v| {
            *v += 1;
        });
        // Use a drop guard so the counter is decremented even if this future
        // is cancelled (e.g. by tokio::select! picking a cancellation branch).
        // Without this, a cancelled decode leaks a +1 in the counter, causing
        // black_box_decoder.reset() to wait forever.
        let mut decoding_guard = DecodingGuard::new(self.decoding.clone());
        let decoding_tx = self.decoding.clone();
        self.thread_pool.spawn(move || {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let hypergraph = problem.hypergraph.as_ref().unwrap();
                let mut instance = T::new(hypergraph, &original_config);
                instance
                    .decode(DecodeRequest {
                        syndrome: problem.syndrome.as_ref().unwrap(),
                        decoder_seed: problem.decoder_seed,
                        reweights: &[],
                        loss: problem.loss.as_ref(),
                    })
                    .and_then(|parity_factor| {
                        validation::validate_parity_factor(parity_factor, hypergraph.hyperedges.len())
                            .map_err(DecodeError::Backend)
                    })
            }));
            match result {
                Ok(decode_result) => {
                    let _ = tx.send(decode_result);
                }
                Err(_) => {
                    eprintln!("decoder panicked during decode");
                }
            }
            decoding_tx.send_modify(|v| {
                *v -= 1;
            });
        });
        // The worker owns the decrement from here. A cancelled RPC drops its
        // receiver, but reset must still wait for the backend work to finish.
        decoding_guard.defuse();
        let parity_factor = rx
            .await
            .map_err(|_| Status::internal("decode panicked or was cancelled"))?
            .map_err(decode_error_status)?;
        Ok(parity_factor.into())
    }

    async fn load_hypergraph(
        &self,
        request: Request<blackbox_decoder::DecodingHypergraph>,
    ) -> Result<Response<blackbox_decoder::LoadHypergraphResponse>, Status> {
        let hypergraph = Arc::new(request.into_inner());
        validation::validate_hypergraph(&hypergraph).map_err(Status::invalid_argument)?;
        T::validate_hypergraph(&hypergraph, &self.original_config).map_err(Status::invalid_argument)?;
        let hid = self.next_hid.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = oneshot::channel::<Result<(T, Arc<DecodingHypergraph>, DecodingGuard), Status>>();
        let original_config = self.original_config.clone();
        self.decoding.send_modify(|count| *count += 1);
        let decoding_guard = DecodingGuard::new(self.decoding.clone());
        self.thread_pool.spawn(move || {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| T::new(&hypergraph, &original_config)));
            let result = match result {
                Ok(instance) => Ok((instance, hypergraph, decoding_guard)),
                Err(_) => {
                    eprintln!("decoder panicked during load_hypergraph (hid={hid})");
                    Err(Status::internal(format!("hid={hid} load panicked")))
                }
            };
            let _ = tx.send(result);
        });
        let (instance, hypergraph, decoding_guard) = rx
            .await
            .map_err(|_| Status::internal(format!("hid={hid} load was cancelled")))??;
        let mut instances = LinkedList::new();
        instances.push_back(instance);
        self.loaded.lock().await.insert(hid, Loaded { hypergraph, instances });
        drop(decoding_guard);
        Ok(Response::new(blackbox_decoder::LoadHypergraphResponse { hid }))
    }

    async fn decode_loaded(
        &self,
        request: Request<blackbox_decoder::LoadedDecodingProblem>,
    ) -> Result<Response<blackbox_decoder::ParityFactor>, Status> {
        let problem = request.into_inner();
        let syndrome = problem
            .syndrome
            .as_ref()
            .ok_or_else(|| Status::invalid_argument("missing syndrome"))?;
        let reweights: Vec<(u64, f64)> = problem
            .reweights
            .iter()
            .map(|value| (value.edge, value.probability))
            .collect();
        let request = DecodeRequest {
            syndrome,
            decoder_seed: problem.decoder_seed,
            reweights: &reweights,
            loss: problem.loss.as_ref(),
        };
        request.require_supported(self.features).map_err(decode_error_status)?;
        // Increment counter BEFORE accessing the loaded map, so that
        // reset() always sees counter > 0 while we're processing.
        self.decoding.send_modify(|v| {
            *v += 1;
        });
        let decoding_guard = DecodingGuard::new(self.decoding.clone());
        let (instance, hypergraph, edge_count) = {
            let mut guard = self.loaded.lock().await;
            let Some(loaded) = guard.get_mut(&problem.hid) else {
                return Err(Status::not_found(format!("hid={}", problem.hid)));
            };
            let edge_count = loaded.hypergraph.hyperedges.len();
            validation::validate_syndrome(syndrome, loaded.hypergraph.vertex_num).map_err(Status::invalid_argument)?;
            validation::validate_reweights(&reweights, edge_count).map_err(Status::invalid_argument)?;
            validation::validate_loss(problem.loss.as_ref(), edge_count).map_err(Status::invalid_argument)?;
            // Validate the handle and side information before applying the
            // plain-request zero-syndrome shortcut.
            if bit_vector::is_zero(syndrome)
                && request.required_features().is_empty()
                && !self.features.contains(DecoderFeatures::OBSERVABLES)
            {
                return Ok(Response::new(ParityFactor { subgraph: vec![] }));
            }
            let instance = loaded.instances.pop_back();
            let hypergraph = instance.is_none().then(|| Arc::clone(&loaded.hypergraph));
            (instance, hypergraph, edge_count)
        };
        let (tx, rx) = oneshot::channel::<Result<(ParityFactor, Option<T>, DecodingGuard), DecodeError>>();
        let original_config = instance.is_none().then(|| Arc::clone(&self.original_config));
        let hid = problem.hid;
        self.thread_pool.spawn(move || {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let mut instance =
                    instance.unwrap_or_else(|| T::new(hypergraph.as_deref().unwrap(), original_config.as_deref().unwrap()));
                let decode_result = instance
                    .decode(DecodeRequest {
                        syndrome: problem.syndrome.as_ref().unwrap(),
                        decoder_seed: problem.decoder_seed,
                        reweights: &reweights,
                        loss: problem.loss.as_ref(),
                    })
                    .and_then(|parity_factor| {
                        validation::validate_parity_factor(parity_factor, edge_count).map_err(DecodeError::Backend)
                    });
                (instance, decode_result)
            }));
            match result {
                Ok((mut instance, Ok(parity_factor))) => {
                    let reset_succeeded =
                        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| instance.reset())).is_ok();
                    if !reset_succeeded {
                        eprintln!("decoder panicked during reset after decode_loaded (hid={hid})");
                    }
                    let instance = reset_succeeded.then_some(instance);
                    let _ = tx.send(Ok((parity_factor, instance, decoding_guard)));
                }
                Ok((_instance, Err(error))) => {
                    let _ = tx.send(Err(error));
                }
                Err(_) => {
                    eprintln!("decoder panicked during decode_loaded (hid={hid})");
                }
            }
        });
        let (parity_factor, instance, decoding_guard) = rx
            .await
            .map_err(|_| Status::internal("decode panicked or was cancelled"))?
            .map_err(decode_error_status)?;
        if let Some(instance) = instance {
            let mut guard = self.loaded.lock().await;
            if let Some(loaded) = guard.get_mut(&hid) {
                loaded.instances.push_back(instance);
            }
        }
        drop(decoding_guard);
        Ok(parity_factor.into())
    }

    async fn reset(&self, request: Request<blackbox_decoder::ResetRequest>) -> Result<Response<()>, Status> {
        let flags = request.into_inner();
        if flags.reset_hypergraphs {
            // Acquire the loaded lock and check the counter atomically.
            // Since decode_loaded increments the counter BEFORE acquiring
            // this lock, seeing counter==0 while holding the lock guarantees
            // no decode_loaded is active or about to start.
            loop {
                let mut loaded = self.loaded.lock().await;
                if *self.decoding.borrow() == 0 {
                    loaded.clear();
                    break;
                }
                // In-flight decodes need the lock to return instances,
                // so drop it and wait for them to finish.
                drop(loaded);
                let mut rx = self.decoding.subscribe();
                rx.wait_for(|v| *v == 0).await.unwrap();
            }
        } else if *self.decoding.borrow() > 0 {
            let mut rx = self.decoding.subscribe();
            rx.wait_for(|v| *v == 0).await.unwrap();
        }
        Ok(().into())
    }
}

fn decode_error_status(error: DecodeError) -> Status {
    match error {
        DecodeError::Unsupported(_) => Status::failed_precondition(error.to_string()),
        DecodeError::InvalidInput(_) => Status::invalid_argument(error.to_string()),
        DecodeError::Backend(_) => Status::internal(error.to_string()),
    }
}
